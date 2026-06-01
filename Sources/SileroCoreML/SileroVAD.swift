import CoreML
import Foundation

public final class SileroVAD: @unchecked Sendable {
    public static let modelVersion = "6.2.1"
    public static let sampleRate = SileroCoreML.sampleRate
    public static let chunkSize = SileroCoreML.chunkSize
    public static let defaultThreshold: Float = 0.5

    private let predictor: any SileroVADPredicting
    private let lock = NSLock()
    private var streamingState: SileroCoreML.StreamingState

    public convenience init(
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        guard let modelURL = Bundle.module.url(forResource: "SileroVAD", withExtension: "mlpackage") else {
            throw SileroVADError.bundledModelMissing("SileroVAD.mlpackage was not found in package resources.")
        }

        try self.init(modelURL: modelURL, configuration: configuration)
    }

    public convenience init(
        modelURL: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) throws {
        try self.init(predictor: CoreMLSileroVADPredictor(modelURL: modelURL, configuration: configuration))
    }

    init(predictor: any SileroVADPredicting) {
        self.predictor = predictor
        streamingState = SileroCoreML.StreamingState()
    }

    public func process(_ chunk: [Float]) throws -> Float {
        lock.lock()
        defer { lock.unlock() }

        do {
            var nextStreamingState = streamingState
            let modelInput = try nextStreamingState.makeModelInput(for: chunk)
            let prediction = try predictor.predict(input: modelInput, state: nextStreamingState.state)
            try nextStreamingState.updateState(with: prediction.nextState)
            streamingState = nextStreamingState
            return prediction.probability
        } catch {
            throw SileroVADError(error)
        }
    }

    public func isSpeech(_ chunk: [Float], threshold: Float = defaultThreshold) throws -> Bool {
        try process(chunk) >= threshold
    }

    public func processStateless(_ chunk: [Float]) throws -> Float {
        lock.lock()
        defer { lock.unlock() }

        do {
            var statelessState = SileroCoreML.StreamingState()
            let modelInput = try statelessState.makeModelInput(for: chunk)
            let prediction = try predictor.predict(input: modelInput, state: statelessState.state)
            return prediction.probability
        } catch {
            throw SileroVADError(error)
        }
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        streamingState.reset()
    }
}

public enum SileroVADError: Error, Equatable, LocalizedError {
    case invalidChunkSize(expected: Int, actual: Int)
    case invalidStateSize(expected: Int, actual: Int)
    case bundledModelMissing(String)
    case modelContractViolation(String)
    case predictionFailed(String)

    init(_ error: Error) {
        switch error {
        case let error as SileroVADError:
            self = error
        case let error as SileroCoreML.StreamingStateError:
            switch error {
            case let .invalidChunkSize(expected, actual):
                self = .invalidChunkSize(expected: expected, actual: actual)
            case let .invalidStateSize(expected, actual):
                self = .invalidStateSize(expected: expected, actual: actual)
            }
        case let error as SileroVADPredictionError:
            self = .modelContractViolation(error.description)
        default:
            self = .predictionFailed(String(describing: error))
        }
    }

    public var errorDescription: String? {
        switch self {
        case let .invalidChunkSize(expected, actual):
            return "Expected chunk of \(expected) samples but received \(actual)."
        case let .invalidStateSize(expected, actual):
            return "Expected flattened state of \(expected) values but received \(actual)."
        case let .bundledModelMissing(message):
            return "Bundled CoreML model missing: \(message)"
        case let .modelContractViolation(message):
            return "CoreML model contract mismatch: \(message)"
        case let .predictionFailed(message):
            return "CoreML prediction failed: \(message)"
        }
    }
}

struct SileroVADPrediction: Equatable {
    let probability: Float
    let nextState: [Float]
}

protocol SileroVADPredicting {
    func predict(input: [Float], state: [Float]) throws -> SileroVADPrediction
}

enum SileroVADPredictionError: Error, Equatable {
    case invalidModelInputSize(expected: Int, actual: Int)
    case invalidStateInputSize(expected: Int, actual: Int)
    case missingOutput(name: String)
    case emptyOutput(name: String)
    case unsupportedOutputType(name: String)
    case missingNextState(name: String)
    case invalidNextStateSize(expected: Int, actual: Int)
}

private extension SileroVADPredictionError {
    var description: String {
        switch self {
        case let .invalidModelInputSize(expected, actual):
            return "Expected model input size \(expected), got \(actual)."
        case let .invalidStateInputSize(expected, actual):
            return "Expected state input size \(expected), got \(actual)."
        case let .missingOutput(name):
            return "Missing output '\(name)'."
        case let .emptyOutput(name):
            return "Output '\(name)' is empty."
        case let .unsupportedOutputType(name):
            return "Output '\(name)' has an unsupported type."
        case let .missingNextState(name):
            return "Missing next state '\(name)'."
        case let .invalidNextStateSize(expected, actual):
            return "Expected next state size \(expected), got \(actual)."
        }
    }
}

private final class CoreMLSileroVADPredictor: SileroVADPredicting {
    private let model: MLModel

    init(modelURL: URL, configuration: MLModelConfiguration) throws {
        let loadableURL = try Self.loadableModelURL(for: modelURL)
        model = try MLModel(contentsOf: loadableURL, configuration: configuration)
    }

    private static func loadableModelURL(for modelURL: URL) throws -> URL {
        switch modelURL.pathExtension {
        case "mlmodel", "mlpackage":
            return try MLModel.compileModel(at: modelURL)
        default:
            return modelURL
        }
    }

    func predict(input: [Float], state: [Float]) throws -> SileroVADPrediction {
        guard input.count == SileroCoreML.modelInputSize else {
            throw SileroVADPredictionError.invalidModelInputSize(
                expected: SileroCoreML.modelInputSize,
                actual: input.count
            )
        }
        guard state.count == SileroCoreML.stateSize else {
            throw SileroVADPredictionError.invalidStateInputSize(
                expected: SileroCoreML.stateSize,
                actual: state.count
            )
        }

        let inputArray = try Self.multiArray(
            from: input,
            shape: [1, NSNumber(value: SileroCoreML.modelInputSize)]
        )
        let stateArray = try Self.multiArray(
            from: state,
            shape: SileroCoreML.stateShape.map(NSNumber.init(value:))
        )
        let features = try MLDictionaryFeatureProvider(dictionary: [
            SileroCoreML.inputName: MLFeatureValue(multiArray: inputArray),
            SileroCoreML.stateName: MLFeatureValue(multiArray: stateArray),
        ])

        let outputFeatures = try model.prediction(from: features)
        let probability = try Self.probability(from: outputFeatures)
        let nextState = try Self.nextState(from: outputFeatures)
        return SileroVADPrediction(probability: probability, nextState: nextState)
    }

    private static func multiArray(from values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    private static func probability(from features: MLFeatureProvider) throws -> Float {
        guard let output = features.featureValue(for: SileroCoreML.outputName) else {
            throw SileroVADPredictionError.missingOutput(name: SileroCoreML.outputName)
        }

        switch output.type {
        case .double:
            return Float(output.doubleValue)
        case .int64:
            return Float(output.int64Value)
        case .multiArray:
            guard let array = output.multiArrayValue else {
                throw SileroVADPredictionError.missingOutput(name: SileroCoreML.outputName)
            }
            guard array.count > 0 else {
                throw SileroVADPredictionError.emptyOutput(name: SileroCoreML.outputName)
            }
            return array[0].floatValue
        default:
            throw SileroVADPredictionError.unsupportedOutputType(name: SileroCoreML.outputName)
        }
    }

    private static func nextState(from features: MLFeatureProvider) throws -> [Float] {
        guard let nextState = features.featureValue(for: SileroCoreML.nextStateName)?.multiArrayValue else {
            throw SileroVADPredictionError.missingNextState(name: SileroCoreML.nextStateName)
        }
        guard nextState.count == SileroCoreML.stateSize else {
            throw SileroVADPredictionError.invalidNextStateSize(
                expected: SileroCoreML.stateSize,
                actual: nextState.count
            )
        }

        return (0..<nextState.count).map { nextState[$0].floatValue }
    }
}
