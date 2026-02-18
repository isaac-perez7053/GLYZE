# @staticmethod
# def random_intersterification(g_composition: GlycerideMix):
#     """
#     Perform intersterification among a list of glycerides.

#     Args:
#         glycerides (list of Glyceride): List of glycerides to undergo intersterification.

#     Returns:
#         list of Glyceride: The resulting glycerides after intersterification.
#     """
#     # Random Interesterification model
#     g_grid = np.zeros((21, 21, 21))

#     # Populate the grid with glyceride chain lengths
#     for glyceride, qty in g_composition.components.items():
#         chain_lengths = glyceride.chain_lengths()
#         # Unpack chain lengths and increment the grid
#         g_grid[*chain_lengths] = qty

#     # Find the unique fatty acid chain lengths present
#     unique_indices_list = np.unique(np.concatenate(np.nonzero(g_grid))).tolist()

#     # Generate output by calculating number of fatty acid with each length present
#     output = []
#     for i in range(0, len(unique_indices_list)):
#         # Sum over the grid to count occurrences of each fatty acid length
#         output.append(
#             1
#             / 3
#             * sum(
#                 sum(g_grid[unique_indices_list[i], :, :]),
#                 sum(g_grid[:, unique_indices_list[i], :]),
#                 sum(g_grid[:, :, unique_indices_list[i]]),
#             )
#         )
#     return output
