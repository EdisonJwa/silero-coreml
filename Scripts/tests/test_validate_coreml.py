from __future__ import annotations

import sys
import unittest
import importlib
from pathlib import Path
from typing import Any, cast


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

validate = importlib.import_module("validate_coreml")


class FakeArray:
    def __init__(self, data: list[float], shape: tuple[int, ...]) -> None:
        self.data = list(data)
        self.shape = shape

    @property
    def size(self) -> int:
        return len(self.data)

    def reshape(self, *shape: Any) -> "FakeArray":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            next_shape = cast(tuple[int, ...], shape[0])
        else:
            next_shape = tuple(int(dimension) for dimension in shape)
        if next_shape == (-1,):
            next_shape = (len(self.data),)
        expected_size = 1
        for dimension in next_shape:
            expected_size *= dimension
        if expected_size != len(self.data):
            raise ValueError(f"cannot reshape {len(self.data)} values into {next_shape}")
        return FakeArray(self.data, next_shape)

    def astype(self, _dtype: object) -> "FakeArray":
        return FakeArray(self.data, self.shape)

    def copy(self) -> "FakeArray":
        return FakeArray(self.data, self.shape)

    def __getitem__(self, index: int | slice) -> "float | FakeArray":
        if isinstance(index, slice):
            return FakeArray(self.data[index], (len(self.data[index]),))
        return self.data[index]


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(value: object, dtype: object | None = None) -> FakeArray:
        if isinstance(value, FakeArray):
            return value.astype(dtype)
        if isinstance(value, list):
            return FakeArray([float(item) for item in value], (len(value),))
        if isinstance(value, tuple):
            return FakeArray([float(item) for item in value], (len(value),))
        return FakeArray([float(cast(Any, value))], (1,))

    @staticmethod
    def zeros(shape: tuple[int, ...] | int, dtype: object | None = None) -> FakeArray:
        if isinstance(shape, int):
            normalized_shape = (shape,)
        else:
            normalized_shape = shape
        size = 1
        for dimension in normalized_shape:
            size *= dimension
        return FakeArray([0.0] * size, normalized_shape)

    @staticmethod
    def concatenate(arrays: list[FakeArray], axis: int = 0) -> FakeArray:
        if axis != 0:
            raise NotImplementedError("fake concatenate only supports axis 0")
        data: list[float] = []
        for array in arrays:
            data.extend(array.data)
        return FakeArray(data, (len(data),))


class FakeMultiArrayType:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeFeatureType:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.multiArrayType = FakeMultiArrayType(shape)


class FakeFeature:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.type = FakeFeatureType(shape)


class FakeDescription:
    def __init__(
        self,
        inputs: list[tuple[str, tuple[int, ...]]],
        outputs: list[tuple[str, tuple[int, ...]]],
    ) -> None:
        self.input = [FakeFeature(name, shape) for name, shape in inputs]
        self.output = [FakeFeature(name, shape) for name, shape in outputs]


class FakeSpec:
    def __init__(
        self,
        inputs: list[tuple[str, tuple[int, ...]]],
        outputs: list[tuple[str, tuple[int, ...]]],
    ) -> None:
        self.description = FakeDescription(inputs, outputs)


class FakeCoreMLModel:
    def __init__(
        self,
        inputs: list[tuple[str, tuple[int, ...]]],
        outputs: list[tuple[str, tuple[int, ...]]],
    ) -> None:
        self._spec = FakeSpec(inputs, outputs)

    def get_spec(self) -> FakeSpec:
        return self._spec


class RecordingCoreMLModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, FakeArray]] = []

    def predict(self, inputs: dict[str, FakeArray]) -> dict[str, FakeArray]:
        self.calls.append(inputs)
        next_state_value = float(len(self.calls))
        return {
            "output": FakeArray([0.25], (1, 1)),
            "stateN": FakeArray([next_state_value] * 256, validate.STATE_SHAPE),
        }


class ValidateCoreMLTests(unittest.TestCase):
    def test_signature_accepts_final_contract(self) -> None:
        model = FakeCoreMLModel(
            inputs=[
                ("input", (1, 576)),
                ("state_in", (2, 1, 128)),
            ],
            outputs=[
                ("output", (1, 1)),
                ("stateN", (2, 1, 128)),
            ],
        )

        validate.validate_coreml_signature(model)

    def test_signature_rejects_placeholder_input_shape(self) -> None:
        model = FakeCoreMLModel(
            inputs=[
                ("input", (1, 512)),
                ("state_in", (2, 1, 128)),
            ],
            outputs=[
                ("output", (1, 1)),
                ("stateN", (2, 1, 128)),
            ],
        )

        with self.assertRaisesRegex(validate.ValidationError, r"input.*\[1, 576\]"):
            validate.validate_coreml_signature(model)

    def test_signature_rejects_old_interface_names(self) -> None:
        model = FakeCoreMLModel(
            inputs=[
                ("audio_chunk", (1, 512)),
                ("state_in", (2, 1, 128)),
            ],
            outputs=[
                ("speech_probability", (1, 1)),
                ("state_out", (2, 1, 128)),
            ],
        )

        with self.assertRaisesRegex(validate.ValidationError, "missing inputs: input"):
            validate.validate_coreml_signature(model)

    def test_predict_coreml_uses_host_context_and_carries_state(self) -> None:
        model = RecordingCoreMLModel()
        state = validate.zero_state(FakeNumpy)
        context = validate.zero_context(FakeNumpy)
        first_chunk = FakeArray([float(index) for index in range(512)], (512,))
        second_chunk = FakeArray([1000.0 + float(index) for index in range(512)], (512,))

        first_probability, state, context = validate.predict_coreml(
            model,
            FakeNumpy,
            first_chunk,
            state,
            context,
        )
        second_probability, state, context = validate.predict_coreml(
            model,
            FakeNumpy,
            second_chunk,
            state,
            context,
        )

        self.assertEqual(first_probability, 0.25)
        self.assertEqual(second_probability, 0.25)
        self.assertEqual(len(model.calls), 2)

        first_inputs = model.calls[0]
        self.assertEqual(sorted(first_inputs.keys()), ["input", "state_in"])
        self.assertEqual(first_inputs["input"].shape, (1, 576))
        self.assertEqual(first_inputs["state_in"].shape, validate.STATE_SHAPE)
        self.assertEqual(first_inputs["input"].data[:64], [0.0] * 64)
        self.assertEqual(first_inputs["input"].data[64:], first_chunk.data)

        second_inputs = model.calls[1]
        self.assertEqual(second_inputs["input"].shape, (1, 576))
        self.assertEqual(second_inputs["state_in"].data, [1.0] * 256)
        self.assertEqual(second_inputs["input"].data[:64], first_chunk.data[-64:])
        self.assertEqual(second_inputs["input"].data[64:], second_chunk.data)
        self.assertEqual(state.data, [2.0] * 256)
        self.assertIsInstance(context, FakeArray)
        self.assertEqual(cast(FakeArray, context).data, second_chunk.data[-64:])

    def test_reset_semantics_zero_state_and_context(self) -> None:
        model = RecordingCoreMLModel()
        chunk = FakeArray([float(index) for index in range(512)], (512,))

        validate.predict_coreml(
            model,
            FakeNumpy,
            chunk,
            validate.zero_state(FakeNumpy),
            validate.zero_context(FakeNumpy),
        )
        validate.predict_coreml(
            model,
            FakeNumpy,
            chunk,
            validate.zero_state(FakeNumpy),
            validate.zero_context(FakeNumpy),
        )

        for call in model.calls:
            self.assertEqual(call["state_in"].data, [0.0] * 256)
            self.assertEqual(call["input"].data[:64], [0.0] * 64)


if __name__ == "__main__":
    unittest.main()
