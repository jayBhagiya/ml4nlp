import unittest

import torch

from graph_transformer.model import (
    GraphTransformer,
    MultiHeadGraphAttention,
    complete_edge_index,
    count_parameters,
)


class GraphTransformerTest(unittest.TestCase):
    def test_attention_normalizes_over_each_target_neighborhood(self):
        layer = MultiHeadGraphAttention(hidden_dim=8, heads=2)
        features = torch.randn(4, 8)
        edges = torch.tensor(
            [[0, 2, 1, 3, 0, 1], [1, 1, 2, 2, 3, 3]], dtype=torch.long
        )

        _, attention = layer(features, edges)

        for target in edges[1].unique():
            torch.testing.assert_close(
                attention[edges[1] == target].sum(0), torch.ones(2)
            )

    def test_complete_edges_do_not_cross_graphs_or_include_loops(self):
        batch = torch.tensor([0, 0, 0, 1, 1])

        edges = complete_edge_index(batch)

        self.assertEqual(edges.shape, (2, 8))
        self.assertTrue(torch.all(edges[0] != edges[1]))
        self.assertTrue(torch.all(batch[edges[0]] == batch[edges[1]]))

    def test_node_order_permutation_only_permutes_outputs(self):
        torch.manual_seed(7)
        model = GraphTransformer(
            input_dim=3,
            hidden_dim=8,
            heads=2,
            layers=2,
            use_laplacian_pe=False,
        ).eval()
        features = torch.randn(5, 3)
        edges = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]]
        )
        batch = torch.zeros(5, dtype=torch.long)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())

        expected = model(features, edges, batch)[permutation]
        actual = model(features[permutation], inverse[edges], batch)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_paper_configuration_keeps_published_parameter_budget(self):
        model = GraphTransformer()
        self.assertEqual(count_parameters(model), 522_982)


if __name__ == "__main__":
    unittest.main()

