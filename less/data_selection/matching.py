import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{}-ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The list of training dataset names')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="The list of checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="The list of checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The list of target task names")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{}-ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="../selected_data",
                       help='The path to the output')
argparser.add_argument(
        "--mode", type=str, choices=["selected_data", "attribution_matrix", "most_influential", "least_influential"],
        help="the mode that determines shape and usage of the final influence matrix"
    )


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALIDATION x N_DIM
    """
    # return an N x N_VALIDATION matrix (2-D tensor)
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task, on each training dataset
for target_task_name in args.target_task_names:
    output_dir = os.path.join(args.output_path, target_task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for train_file_name in args.train_file_names:
        influence_score = 0
        for i, ckpt in enumerate(args.ckpts):
            validation_path = args.validation_gradient_path.format(
            target_task_name, ckpt)
            # validation_path = args.validation_gradient_path.format(
            #     ckpt, target_task_name)
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            validation_info = torch.load(validation_path)

            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()

            gradient_path = args.gradient_path.format(train_file_name, ckpt)
            # gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            training_info = torch.load(gradient_path)

            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()

            # sum influence scores across different epochs (i.e., ckpts), by different ckpt weights
            # N * N_VALIDATION; This is exactly the "most original" attribution matrix that Hao wants!
            influence_score += args.checkpoint_weights[i] * \
                calculate_influence_score(
                    training_info=training_info, validation_info=validation_info)

        print(f"Influence Score shape for {train_file_name} * {target_task_name}: {influence_score.shape}")

        if args.mode == "attribution_matrix":
            # get one row in the attribution matrix: 1 * M_subtasks
            influence_score = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[target_task_name], -1
            ).mean(-1).mean(0, keepdim=True)
            assert influence_score.shape == torch.Size([1, N_SUBTASKS[target_task_name]]), f"{influence_score}, {influence_score.shape}"
            output_file = os.path.join(args.output_path, target_task_name, f"{train_file_name}_attribution_matrix.pt")
            print("Saved attribution matrix to {}".format(output_file))

        elif args.mode == "selected_data":
            # get the column-max influence scores for training point selection
            influence_score = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[target_task_name], -1
            ).mean(-1).max(-1)[0] # first take an average over all eval points inside each subtask; then pick the max influence score over all subtasks as the final influence
            output_file = os.path.join(args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
            print("Saved influence scores to {}".format(output_file))

        elif args.mode == "most_influential":
            # get the row_max influence scores and the corresponding training point idx
            influence_score, indices = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[target_task_name], -1
            ).mean(-1).max(0) # first take an average over all eval points inside each subtask; then pick the training point that is most influential for each subtask
            print(influence_score, influence_score.shape)
            print(indices, indices.shape)
            assert influence_score.shape == torch.Size([N_SUBTASKS[target_task_name]]), f"{influence_score}, {influence_score.shape}"
            assert indices.shape == torch.Size([N_SUBTASKS[target_task_name]]), f"{indices}, {indices.shape}"

            output_file = os.path.join(args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
            print("Saved influence scores to {}".format(output_file))
            outpath_indices = os.path.join(args.output_path, target_task_name, f"{train_file_name}_most_influential_indices.pt")
            print("Saved indices of most influential training points to {}".format(outpath_indices))
            torch.save(indices, outpath_indices)

        elif args.mode == "least_influential":
            # get the row_min influence scores and the corresponding training point idx
            influence_score, indices = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[target_task_name], -1
            ).mean(-1).min(0) # first take an average over all eval points inside each subtask; then pick the training point that is least influential for each subtask
            print(influence_score, influence_score.shape)
            print(indices, indices.shape)
            assert influence_score.shape == torch.Size([N_SUBTASKS[target_task_name]]), f"{influence_score}, {influence_score.shape}"
            assert indices.shape == torch.Size([N_SUBTASKS[target_task_name]]), f"{indices}, {indices.shape}"

            output_file = os.path.join(args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
            print("Saved influence scores to {}".format(output_file))
            outpath_indices = os.path.join(args.output_path, target_task_name, f"{train_file_name}_least_influential_indices.pt")
            print("Saved indices of least influential training points to {}".format(outpath_indices))
            torch.save(indices, outpath_indices)

        else:
            print("Mode: ", args.mode)
            raise NotImplementedError

        torch.save(influence_score, output_file)
        
