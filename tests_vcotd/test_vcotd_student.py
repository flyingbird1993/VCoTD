from __future__ import annotations

import types
import unittest

import torch
import torch.nn.functional as F

from model.vcotd_student import VCoTDStudentModel, build_student_from_checkpoint


def tiny_model(**overrides) -> VCoTDStudentModel:
    config = {
        "d_vlm": 4,
        "d": 4,
        "K": 2,
        "h": 2,
        "d_ff": 8,
        "T_obs": 2,
        "T_pred": 2,
        "d_h": 3,
        "dropout": 0.0,
    }
    config.update(overrides)
    return VCoTDStudentModel(**config)


class VCoTDStudentTest(unittest.TestCase):
    def test_gap_ignores_padded_teacher_tokens(self) -> None:
        model = tiny_model()
        features = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [99.0] * 4]]
        )
        mask = torch.tensor([[True, True, False]])
        pooled = model._prepare_visual_tokens(features, mask)
        expected = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
        self.assertTrue(torch.equal(pooled, expected))

    def test_all_student_tokens_contribute_to_hierarchical_losses(self) -> None:
        model = tiny_model()
        with torch.no_grad():
            model.proj_global.weight.copy_(torch.eye(4))
            model.proj_global.bias.zero_()
            model.proj_dim.weight.copy_(torch.eye(4))
            model.proj_dim.bias.zero_()

        teacher = {name: torch.zeros(1, 3, 4) for name in ("shallow", "middle", "deep")}
        base = {name: torch.zeros(1, 3, 4) for name in teacher}
        changed = {name: value.clone() for name, value in base.items()}
        for value in changed.values():
            value[:, -1] = 1.0  # a motion slot, not the leading visual slot

        base_terms = model.compute_distillation_terms(base, teacher)
        changed_terms = model.compute_distillation_terms(changed, teacher)
        for name in ("L_global", "L_spatial", "L_detail"):
            self.assertEqual(float(base_terms[name]), 0.0)
            self.assertGreater(float(changed_terms[name]), 0.0, name)
        self.assertAlmostEqual(float(changed_terms["L_global"]), 4.0 / 9.0, places=6)

    def test_spatial_loss_interpolates_before_normalizing(self) -> None:
        model = tiny_model()
        student_energy = torch.tensor([[1.0, 4.0, 9.0]])
        aligned = F.interpolate(
            student_energy.unsqueeze(1), size=4, mode="linear", align_corners=False
        ).squeeze(1)
        teacher_middle = torch.zeros(1, 4, 4)
        teacher_middle[:, :, 0] = aligned.sqrt()
        student_middle = torch.zeros(1, 3, 4)
        student_middle[:, :, 0] = student_energy.sqrt()
        teacher = {
            "shallow": torch.zeros(1, 4, 4),
            "middle": teacher_middle,
            "deep": torch.zeros(1, 4, 4),
        }
        student = {
            "shallow": torch.zeros(1, 3, 4),
            "middle": student_middle,
            "deep": torch.zeros(1, 3, 4),
        }
        spatial = model.compute_distillation_terms(student, teacher)["L_spatial"]
        self.assertTrue(torch.allclose(spatial, torch.zeros_like(spatial), atol=1e-7))

    def test_teacher_padding_does_not_change_distillation_terms(self) -> None:
        torch.manual_seed(7)
        model = tiny_model()
        student = {name: torch.randn(1, 3, 4) for name in ("shallow", "middle", "deep")}
        teacher = {name: torch.randn(1, 4, 4) for name in student}
        mask = torch.tensor([[True, True, True, False]])
        altered = {name: value.clone() for name, value in teacher.items()}
        for value in altered.values():
            value[:, -1] = 10000.0

        original = model.compute_distillation_terms(student, teacher, mask)
        padded = model.compute_distillation_terms(student, altered, mask)
        for name in original:
            self.assertTrue(torch.allclose(original[name], padded[name]), name)

    def test_adaptive_alpha_is_applied_per_sample(self) -> None:
        model = tiny_model(alpha_min=0.3, alpha_max=1.0)
        terms = {
            "L_global": torch.tensor([2.0, 4.0]),
            "L_spatial": torch.zeros(2),
            "L_detail": torch.zeros(2),
        }

        def fixed_terms(_self, *_args, **_kwargs):
            return terms

        model.compute_distillation_terms = types.MethodType(fixed_terms, model)
        prediction = torch.zeros(2, 2, 2)
        target = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]]
        )
        gamma = torch.tensor([[0.0], [1.0]])
        total, metrics = model.compute_total_loss(
            prediction, target, {}, {}, gamma, adaptive=True
        )
        self.assertAlmostEqual(float(total), 3.8, places=6)
        self.assertAlmostEqual(metrics["alpha"], 0.65, places=6)

    def test_fixed_alpha_ablation(self) -> None:
        model = tiny_model()
        terms = {
            "L_global": torch.tensor([2.0, 4.0]),
            "L_spatial": torch.zeros(2),
            "L_detail": torch.zeros(2),
        }

        def fixed_terms(_self, *_args, **_kwargs):
            return terms

        model.compute_distillation_terms = types.MethodType(fixed_terms, model)
        prediction = torch.zeros(2, 2, 2)
        target = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]]
        )
        total, metrics = model.compute_total_loss(
            prediction,
            target,
            {},
            {},
            torch.zeros(2, 1),
            adaptive=False,
            fixed_alpha=0.5,
        )
        self.assertAlmostEqual(float(total), 3.0, places=6)
        self.assertAlmostEqual(metrics["alpha"], 0.5, places=6)

    def test_trajectory_loss_ignores_invalid_horizons(self) -> None:
        prediction = torch.tensor([[[3.0, 4.0], [100.0, 0.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False]])
        loss = VCoTDStudentModel.trajectory_loss_per_sample(prediction, target, mask)
        self.assertEqual(loss.tolist(), [5.0])

    def test_predict_skips_training_only_complexity_estimator(self) -> None:
        model = tiny_model().eval()
        calls = []
        handle = model.complexity_estimator.register_forward_hook(
            lambda *_args: calls.append(True)
        )
        try:
            model.predict(torch.randn(1, 3, 4), torch.randn(1, 2, 2))
            self.assertEqual(calls, [])
            model(torch.randn(1, 3, 4), torch.randn(1, 2, 2))
            self.assertEqual(calls, [True])
        finally:
            handle.remove()

    def test_legacy_checkpoint_shape_inference(self) -> None:
        for visual_tokens in (1, 3):
            original = VCoTDStudentModel(
                d_vlm=8,
                d=8,
                K=2,
                h=4,
                d_ff=16,
                T_obs=6,
                T_pred=3,
                d_h=5,
                dropout=0.0,
                visual_aggregation="gap" if visual_tokens == 1 else "uniform",
                vis_tokens=visual_tokens,
            )
            restored = build_student_from_checkpoint(
                {"model_state": original.state_dict()}
            )
            self.assertEqual(restored.visual_aggregation, "uniform")
            self.assertEqual(restored.vis_tokens, visual_tokens)
            self.assertEqual(restored.config.d_ff, 16)
            self.assertEqual(restored.T_pred, 3)

        current = VCoTDStudentModel(
            d_vlm=8,
            d=8,
            K=2,
            h=4,
            d_ff=16,
            T_obs=6,
            T_pred=3,
            d_h=5,
            dropout=0.0,
            visual_aggregation="gap",
            vis_tokens=1,
        )
        restored = build_student_from_checkpoint(
            {
                "model_state": current.state_dict(),
                "model_config": current.model_config(),
            }
        )
        self.assertEqual(restored.visual_aggregation, "gap")


if __name__ == "__main__":
    unittest.main()
