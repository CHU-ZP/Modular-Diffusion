import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from diffusion.builders import (
    build_autoencoder,
    build_loss,
    build_model,
    build_parameterization,
    build_process,
    build_representation,
    build_sampler,
    build_schedule,
)
from diffusion.data.cifar10 import (
    HuggingFaceCIFAR10Dataset,
    build_cifar10_dataloader,
)
from diffusion.devices import resolve_device
from diffusion.parameterizations import DiffusionParameterization
from diffusion.processes import DiffusionProcess
from diffusion.representations import LatentRepresentation, PixelRepresentation
from diffusion.schedules import make_schedule
from diffusion.train import build_dataloader


class TinyAutoencoder(nn.Module):
    def encode(self, image):
        return image[:, :1]

    def decode(self, latent):
        return latent.repeat(1, 3, 1, 1)


class DiffusionSmokeTests(unittest.TestCase):
    def test_schedule_shapes(self):
        for schedule_type in ("linear", "cosine", "sigmoid"):
            schedule = make_schedule(schedule_type, num_timesteps=16)
            self.assertEqual(schedule.betas.shape, (16,))
            self.assertEqual(schedule.alpha_bars.shape, (16,))
            self.assertTrue(torch.isfinite(schedule.snr).all())

    def test_q_sample_shape(self):
        schedule = make_schedule("linear", num_timesteps=16)
        process = DiffusionProcess(schedule)
        x0 = torch.randn(4, 1, 8, 8)
        noise = torch.randn_like(x0)
        timesteps = torch.tensor([0, 1, 2, 3])
        x_t = process.q_sample(x0, timesteps, noise)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertTrue(torch.isfinite(x_t).all())

    def test_prediction_target_roundtrip(self):
        schedule = make_schedule("linear", num_timesteps=16)
        process = DiffusionProcess(schedule)
        x0 = torch.randn(4, 1, 8, 8)
        noise = torch.randn_like(x0)
        timesteps = torch.tensor([1, 3, 5, 7])
        x_t = process.q_sample(x0, timesteps, noise)

        for target in ("epsilon", "x0", "v"):
            parameterization = DiffusionParameterization(schedule, target)
            model_output = parameterization.target_from(x0, noise, x_t, timesteps)
            pred_noise = parameterization.model_output_to_epsilon(model_output, x_t, timesteps)
            pred_x0 = parameterization.model_output_to_x0(model_output, x_t, timesteps)
            self.assertTrue(torch.allclose(pred_noise, noise, atol=1e-5))
            self.assertTrue(torch.allclose(pred_x0, x0, atol=1e-5))

    def test_model_forward_and_loss_backward(self):
        config = {
            "data": {"image_shape": [1, 8, 8]},
            "model": {
                "type": "mlp",
                "input_shape": [1, 8, 8],
                "hidden_dims": [32],
                "time_embedding_dim": 16,
            },
            "schedule": {"type": "linear", "num_timesteps": 8},
            "diffusion": {"prediction_target": "epsilon"},
            "loss": {"type": "mse"},
        }
        schedule = build_schedule(config)
        process = build_process(schedule)
        parameterization = build_parameterization(config, schedule)
        model = build_model(config)
        loss_fn = build_loss(config, process, parameterization)

        x0 = torch.randn(2, 1, 8, 8)
        timesteps = torch.tensor([1, 2])
        output = model(x0, timesteps)
        self.assertEqual(output.shape, x0.shape)
        loss = loss_fn(model, x0)
        loss.backward()
        total_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total_grad, 0.0)

    def test_dit_forward_shape(self):
        config = {
            "data": {"image_shape": [1, 8, 8]},
            "model": {
                "type": "dit",
                "input_shape": [1, 8, 8],
                "patch_size": 4,
                "embed_dim": 32,
                "depth": 2,
                "num_heads": 4,
            },
        }
        model = build_model(config)
        x_t = torch.randn(2, 1, 8, 8)
        timesteps = torch.tensor([1, 2])
        output = model(x_t, timesteps)
        self.assertEqual(output.shape, x_t.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_samplers_generate_finite_values_for_all_targets(self):
        for target in ("epsilon", "x0", "v"):
            config = {
                "data": {"image_shape": [1, 8, 8]},
                "model": {
                    "type": "mlp",
                    "input_shape": [1, 8, 8],
                    "hidden_dims": [16],
                    "time_embedding_dim": 16,
                },
                "schedule": {"type": "linear", "num_timesteps": 4},
                "diffusion": {"prediction_target": target},
            }
            schedule = build_schedule(config)
            process = build_process(schedule)
            parameterization = build_parameterization(config, schedule)
            model = build_model(config)
            for sampler_config in (
                {"type": "ddpm"},
                {"type": "ddim", "num_steps": 2, "eta": 0.0},
            ):
                sampler = build_sampler({"sampler": sampler_config}, process, parameterization)
                sample = sampler.sample(model, shape=(2, 1, 8, 8), device="cpu")
                self.assertEqual(sample.shape, (2, 1, 8, 8))
                self.assertTrue(torch.isfinite(sample).all())

    def test_representations(self):
        image = torch.randn(2, 3, 8, 8)
        pixel = PixelRepresentation()
        self.assertTrue(torch.equal(pixel.decode(pixel.encode(image)), image))

        latent = LatentRepresentation(TinyAutoencoder(), scaling_factor=2.0)
        clean = latent.encode(image)
        decoded = latent.decode(clean)
        self.assertEqual(clean.shape, (2, 1, 8, 8))
        self.assertEqual(decoded.shape, image.shape)

    def test_latent_representation_can_build_default_autoencoder(self):
        config = {
            "data": {"image_shape": [3, 8, 8]},
            "model": {"type": "unet", "in_channels": 4, "base_channels": 8},
            "representation": {
                "type": "latent",
                "scaling_factor": 0.5,
                "autoencoder": {
                    "type": "conv",
                    "hidden_channels": 8,
                    "num_res_blocks": 1,
                    "downsample_factor": 1,
                },
            },
        }
        representation = build_representation(config)
        image = torch.randn(2, 3, 8, 8)
        clean = representation.encode(image)
        decoded = representation.decode(clean)
        self.assertEqual(clean.shape, (2, 4, 8, 8))
        self.assertEqual(decoded.shape, image.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_diffusers_autoencoder_config_uses_loader(self):
        config = {
            "representation": {
                "type": "latent",
                "autoencoder": {
                    "type": "diffusers_autoencoder_kl",
                    "pretrained_model_name_or_path": "checkpoints/vae/sd-vae-ft-mse",
                    "torch_dtype": "float32",
                    "local_files_only": True,
                },
            },
        }
        with patch("diffusion.builders.load_diffusers_autoencoder_kl") as loader:
            loader.return_value = TinyAutoencoder()
            autoencoder = build_autoencoder(config)
        self.assertIsInstance(autoencoder, TinyAutoencoder)
        loader.assert_called_once_with(
            "checkpoints/vae/sd-vae-ft-mse",
            subfolder=None,
            revision=None,
            variant=None,
            torch_dtype="float32",
            local_files_only=True,
            cache_dir=None,
        )

    def test_build_dataloader_ignores_data_metadata(self):
        config = {
            "data": {
                "type": "cifar10",
                "source": "huggingface",
                "root": "./data",
                "image_shape": [3, 32, 32],
                "num_classes": 10,
                "batch_size": 4,
                "num_workers": 0,
            },
        }
        with patch("diffusion.train.build_cifar10_dataloader") as build_loader:
            build_dataloader(config)
        build_loader.assert_called_once_with(
            source="huggingface",
            root="./data",
            batch_size=4,
            num_workers=0,
        )

    def test_huggingface_cifar10_matches_training_interface(self):
        fake_records = [
            {"img": Image.new("RGB", (32, 32), color=(128, 64, 32)), "label": 7},
        ]
        with patch("diffusion.data.cifar10._load_huggingface_split") as load_split:
            load_split.return_value = fake_records
            dataset = HuggingFaceCIFAR10Dataset(root="./data", train=True)
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertEqual(label, 7)
        self.assertGreaterEqual(float(image.min()), -1.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_huggingface_cifar10_dataloader_batches(self):
        fake_records = [
            {"img": Image.new("RGB", (32, 32), color=(0, 0, 0)), "label": 0},
            {"img": Image.new("RGB", (32, 32), color=(255, 255, 255)), "label": 1},
        ]
        with patch("diffusion.data.cifar10._load_huggingface_split") as load_split:
            load_split.return_value = fake_records
            loader = build_cifar10_dataloader(
                batch_size=2,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
            )
            images, labels = next(iter(loader))
        self.assertEqual(images.shape, (2, 3, 32, 32))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertTrue(torch.isclose(images[0].min(), torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(images[1].max(), torch.tensor(1.0)))

    def test_resolve_device_auto(self):
        self.assertIn(resolve_device("auto").type, {"cpu", "cuda"})
        self.assertIn(resolve_device(None).type, {"cpu", "cuda"})
        self.assertEqual(resolve_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
