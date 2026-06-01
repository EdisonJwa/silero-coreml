import XCTest
@testable import SileroCoreML

final class SileroCoreMLTests: XCTestCase {
    func testBackendIdentifierMatchesChanoraIntegrationName() {
        XCTAssertEqual(VadBackend.SileroCoreMl, "VadBackend::SileroCoreMl")
        XCTAssertEqual(SileroCoreML.backendIdentifier, VadBackend.SileroCoreMl)
    }

    func testContractConstantsMatchPlannedCoreMLTarget() {
        XCTAssertEqual(SileroCoreML.sampleRate, 16_000)
        XCTAssertEqual(SileroCoreML.chunkSize, 512)
        XCTAssertEqual(SileroCoreML.contextSize, 64)
        XCTAssertEqual(SileroCoreML.modelInputSize, 576)
        XCTAssertEqual(SileroCoreML.stateShape, [2, 1, 128])
        XCTAssertEqual(SileroCoreML.stateSize, 256)

        XCTAssertEqual(SileroCoreML.inputName, "input")
        XCTAssertEqual(SileroCoreML.stateName, "state_in")
        XCTAssertEqual(SileroCoreML.outputName, "output")
        XCTAssertEqual(SileroCoreML.nextStateName, "stateN")

        XCTAssertEqual(SileroVAD.modelVersion, "6.2.1")
        XCTAssertEqual(SileroVAD.sampleRate, SileroCoreML.sampleRate)
        XCTAssertEqual(SileroVAD.chunkSize, SileroCoreML.chunkSize)
        XCTAssertEqual(SileroVAD.defaultThreshold, 0.5)
    }

    func testStreamingStateStartsWithZeroContextAndState() {
        let streamingState = SileroCoreML.StreamingState()

        XCTAssertEqual(streamingState.context.count, SileroCoreML.contextSize)
        XCTAssertEqual(streamingState.state.count, SileroCoreML.stateSize)
        XCTAssertEqual(streamingState.context, Array(repeating: 0, count: SileroCoreML.contextSize))
        XCTAssertEqual(streamingState.state, Array(repeating: 0, count: SileroCoreML.stateSize))
    }

    func testMakeModelInputConcatenatesContextAndChunkAndUpdatesContext() throws {
        var streamingState = SileroCoreML.StreamingState()
        let chunk = (0..<SileroCoreML.chunkSize).map(Float.init)

        let modelInput = try streamingState.makeModelInput(for: chunk)

        XCTAssertEqual(modelInput.count, SileroCoreML.modelInputSize)
        XCTAssertEqual(
            Array(modelInput.prefix(SileroCoreML.contextSize)),
            Array(repeating: 0, count: SileroCoreML.contextSize)
        )
        XCTAssertEqual(Array(modelInput.suffix(SileroCoreML.chunkSize)), chunk)
        XCTAssertEqual(
            streamingState.context,
            Array(chunk.suffix(SileroCoreML.contextSize))
        )
    }

    func testResetZerosContextAndStateAfterUse() throws {
        var streamingState = SileroCoreML.StreamingState()
        let chunk = Array(repeating: Float(1), count: SileroCoreML.chunkSize)
        let nextState = Array(repeating: Float(2), count: SileroCoreML.stateSize)

        _ = try streamingState.makeModelInput(for: chunk)
        try streamingState.updateState(with: nextState)
        streamingState.reset()

        XCTAssertEqual(streamingState.context, Array(repeating: 0, count: SileroCoreML.contextSize))
        XCTAssertEqual(streamingState.state, Array(repeating: 0, count: SileroCoreML.stateSize))
    }

    func testMakeModelInputThrowsForInvalidChunkSize() {
        var streamingState = SileroCoreML.StreamingState()

        XCTAssertThrowsError(try streamingState.makeModelInput(for: [Float](repeating: 0, count: 511))) { error in
            XCTAssertEqual(
                error as? SileroCoreML.StreamingStateError,
                .invalidChunkSize(expected: 512, actual: 511)
            )
            XCTAssertEqual(
                (error as? SileroCoreML.StreamingStateError)?.errorDescription,
                "Expected chunk of 512 samples but received 511."
            )
        }
    }

    func testProcessBuildsExplicitStateInputAndCarriesStateAndContext() throws {
        let predictor = FakeSileroVADPredictor(predictions: [
            SileroVADPrediction(probability: 0.25, nextState: Array(repeating: 2, count: SileroCoreML.stateSize)),
            SileroVADPrediction(probability: 0.75, nextState: Array(repeating: 3, count: SileroCoreML.stateSize)),
        ])
        let vad = SileroVAD(predictor: predictor)
        let firstChunk = (0..<SileroCoreML.chunkSize).map(Float.init)
        let secondChunk = (0..<SileroCoreML.chunkSize).map { Float($0 + 1_000) }

        let firstProbability = try vad.process(firstChunk)
        let secondProbability = try vad.process(secondChunk)

        XCTAssertEqual(firstProbability, 0.25)
        XCTAssertEqual(secondProbability, 0.75)
        XCTAssertEqual(predictor.requests.count, 2)
        XCTAssertEqual(
            Array(predictor.requests[0].input.prefix(SileroCoreML.contextSize)),
            Array(repeating: 0, count: SileroCoreML.contextSize)
        )
        XCTAssertEqual(Array(predictor.requests[0].input.suffix(SileroCoreML.chunkSize)), firstChunk)
        XCTAssertEqual(predictor.requests[0].state, Array(repeating: 0, count: SileroCoreML.stateSize))
        XCTAssertEqual(
            Array(predictor.requests[1].input.prefix(SileroCoreML.contextSize)),
            Array(firstChunk.suffix(SileroCoreML.contextSize))
        )
        XCTAssertEqual(Array(predictor.requests[1].input.suffix(SileroCoreML.chunkSize)), secondChunk)
        XCTAssertEqual(predictor.requests[1].state, Array(repeating: 2, count: SileroCoreML.stateSize))
    }

    func testResetZerosFacadeContextAndState() throws {
        let predictor = FakeSileroVADPredictor(predictions: [
            SileroVADPrediction(probability: 0.25, nextState: Array(repeating: 2, count: SileroCoreML.stateSize)),
            SileroVADPrediction(probability: 0.75, nextState: Array(repeating: 3, count: SileroCoreML.stateSize)),
        ])
        let vad = SileroVAD(predictor: predictor)
        let chunk = Array(repeating: Float(1), count: SileroCoreML.chunkSize)

        _ = try vad.process(chunk)
        vad.reset()
        _ = try vad.process(chunk)

        XCTAssertEqual(predictor.requests.count, 2)
        XCTAssertEqual(
            Array(predictor.requests[1].input.prefix(SileroCoreML.contextSize)),
            Array(repeating: 0, count: SileroCoreML.contextSize)
        )
        XCTAssertEqual(predictor.requests[1].state, Array(repeating: 0, count: SileroCoreML.stateSize))
    }

    func testIsSpeechUsesInclusiveDefaultThresholdAndAllowsOverride() throws {
        let predictor = FakeSileroVADPredictor(predictions: [
            SileroVADPrediction(probability: 0.5, nextState: Array(repeating: 0, count: SileroCoreML.stateSize)),
            SileroVADPrediction(probability: 0.4, nextState: Array(repeating: 0, count: SileroCoreML.stateSize)),
        ])
        let vad = SileroVAD(predictor: predictor)
        let chunk = Array(repeating: Float(0), count: SileroCoreML.chunkSize)

        XCTAssertTrue(try vad.isSpeech(chunk))
        XCTAssertTrue(try vad.isSpeech(chunk, threshold: 0.4))
    }

    func testProcessStatelessDoesNotMutateStreamingState() throws {
        let predictor = FakeSileroVADPredictor(predictions: [
            SileroVADPrediction(probability: 0.25, nextState: Array(repeating: 2, count: SileroCoreML.stateSize)),
            SileroVADPrediction(probability: 0.99, nextState: Array(repeating: 9, count: SileroCoreML.stateSize)),
            SileroVADPrediction(probability: 0.75, nextState: Array(repeating: 3, count: SileroCoreML.stateSize)),
        ])
        let vad = SileroVAD(predictor: predictor)
        let firstChunk = (0..<SileroCoreML.chunkSize).map(Float.init)
        let statelessChunk = Array(repeating: Float(9), count: SileroCoreML.chunkSize)
        let secondChunk = Array(repeating: Float(4), count: SileroCoreML.chunkSize)

        _ = try vad.process(firstChunk)
        let statelessProbability = try vad.processStateless(statelessChunk)
        _ = try vad.process(secondChunk)

        XCTAssertEqual(statelessProbability, 0.99)
        XCTAssertEqual(predictor.requests.count, 3)
        XCTAssertEqual(
            Array(predictor.requests[1].input.prefix(SileroCoreML.contextSize)),
            Array(repeating: 0, count: SileroCoreML.contextSize)
        )
        XCTAssertEqual(predictor.requests[1].state, Array(repeating: 0, count: SileroCoreML.stateSize))
        XCTAssertEqual(
            Array(predictor.requests[2].input.prefix(SileroCoreML.contextSize)),
            Array(firstChunk.suffix(SileroCoreML.contextSize))
        )
        XCTAssertEqual(predictor.requests[2].state, Array(repeating: 2, count: SileroCoreML.stateSize))
    }

    func testFacadeThrowsForInvalidChunkSize() {
        let predictor = FakeSileroVADPredictor(predictions: [])
        let vad = SileroVAD(predictor: predictor)

        XCTAssertThrowsError(try vad.process([Float](repeating: 0, count: 511))) { error in
            XCTAssertEqual(
                error as? SileroVADError,
                .invalidChunkSize(expected: SileroCoreML.chunkSize, actual: 511)
            )
        }
        XCTAssertEqual(predictor.requests.count, 0)
    }

    func testSileroVADErrorDescriptions() {
        XCTAssertEqual(
            SileroVADError.invalidChunkSize(expected: 512, actual: 511).errorDescription,
            "Expected chunk of 512 samples but received 511."
        )
        XCTAssertEqual(
            SileroVADError.invalidStateSize(expected: 256, actual: 0).errorDescription,
            "Expected flattened state of 256 values but received 0."
        )
        XCTAssertEqual(
            SileroVADError.modelContractViolation("Missing output 'output'.").errorDescription,
            "CoreML model contract mismatch: Missing output 'output'."
        )
        XCTAssertEqual(
            SileroVADError.predictionFailed("failed").errorDescription,
            "CoreML prediction failed: failed"
        )
    }

    func testProcessMapsInvalidNextStateToPublicError() {
        let predictor = FakeSileroVADPredictor(predictions: [
            SileroVADPrediction(probability: 0.5, nextState: []),
        ])
        let vad = SileroVAD(predictor: predictor)

        XCTAssertThrowsError(try vad.process(Array(repeating: 0, count: SileroCoreML.chunkSize))) { error in
            guard case let .invalidStateSize(expected, actual) = error as? SileroVADError else {
                return XCTFail("Expected invalid state size error, got \(error)")
            }
            XCTAssertEqual(expected, SileroCoreML.stateSize)
            XCTAssertEqual(actual, 0)
        }
    }

    func testProcessPropagatesPredictorErrorsWithoutMutatingStreamingState() throws {
        let predictor = FakeSileroVADPredictor(results: [
            .failure(FakeSileroVADPredictorError.failed),
            .success(SileroVADPrediction(probability: 0.75, nextState: Array(repeating: 3, count: SileroCoreML.stateSize))),
        ])
        let vad = SileroVAD(predictor: predictor)
        let chunk = Array(repeating: Float(1), count: SileroCoreML.chunkSize)

        XCTAssertThrowsError(try vad.process(chunk)) { error in
            guard case let .predictionFailed(description) = error as? SileroVADError else {
                return XCTFail("Expected prediction failure, got \(error)")
            }
            XCTAssertTrue(description.contains("failed"))
        }
        _ = try vad.process(chunk)

        XCTAssertEqual(predictor.requests.count, 2)
        XCTAssertEqual(
            Array(predictor.requests[1].input.prefix(SileroCoreML.contextSize)),
            Array(repeating: 0, count: SileroCoreML.contextSize)
        )
        XCTAssertEqual(predictor.requests[1].state, Array(repeating: 0, count: SileroCoreML.stateSize))
    }

    func testResetWaitsForInFlightProcess() throws {
        let predictor = BlockingSileroVADPredictor(
            prediction: SileroVADPrediction(
                probability: 0.5,
                nextState: Array(repeating: 7, count: SileroCoreML.stateSize)
            )
        )
        let vad = SileroVAD(predictor: predictor)
        let resetReturned = DispatchSemaphore(value: 0)

        DispatchQueue.global().async {
            _ = try? vad.process(Array(repeating: 1, count: SileroCoreML.chunkSize))
        }
        predictor.waitUntilBlocked()

        DispatchQueue.global().async {
            vad.reset()
            resetReturned.signal()
        }

        XCTAssertEqual(resetReturned.wait(timeout: .now() + 0.05), .timedOut)
        predictor.release()
        XCTAssertEqual(resetReturned.wait(timeout: .now() + 1), .success)

        XCTAssertEqual(predictor.requests.count, 1)
    }

    func testOptionalLocalModelURLSmoke() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["SILERO_COREML_MODEL_PATH"] else {
            throw XCTSkip("Set SILERO_COREML_MODEL_PATH to run the local CoreML smoke test.")
        }

        let vad = try SileroVAD(modelURL: URL(fileURLWithPath: modelPath))
        let probability = try vad.process(Array(repeating: 0, count: SileroCoreML.chunkSize))

        XCTAssertGreaterThanOrEqual(probability, 0)
        XCTAssertLessThanOrEqual(probability, 1)
    }

    func testBundledModelInitializerSmoke() throws {
        let vad = try SileroVAD()
        let probability = try vad.process(Array(repeating: 0, count: SileroCoreML.chunkSize))

        XCTAssertGreaterThanOrEqual(probability, 0)
        XCTAssertLessThanOrEqual(probability, 1)
    }
}

private final class BlockingSileroVADPredictor: SileroVADPredicting {
    struct Request {
        let input: [Float]
        let state: [Float]
    }

    private let prediction: SileroVADPrediction
    private let started = DispatchSemaphore(value: 0)
    private let releaseSemaphore = DispatchSemaphore(value: 0)
    private(set) var requests: [Request] = []

    init(prediction: SileroVADPrediction) {
        self.prediction = prediction
    }

    func predict(input: [Float], state: [Float]) throws -> SileroVADPrediction {
        requests.append(Request(input: input, state: state))
        started.signal()
        releaseSemaphore.wait()
        return prediction
    }

    func waitUntilBlocked() {
        started.wait()
    }

    func release() {
        releaseSemaphore.signal()
    }
}

private enum FakeSileroVADPredictorError: Error, Equatable {
    case failed
    case missingPrediction
}

private final class FakeSileroVADPredictor: SileroVADPredicting {
    struct Request {
        let input: [Float]
        let state: [Float]
    }

    private var results: [Result<SileroVADPrediction, Error>]
    private(set) var requests: [Request] = []

    convenience init(predictions: [SileroVADPrediction]) {
        self.init(results: predictions.map(Result.success))
    }

    init(results: [Result<SileroVADPrediction, Error>]) {
        self.results = results
    }

    func predict(input: [Float], state: [Float]) throws -> SileroVADPrediction {
        requests.append(Request(input: input, state: state))
        guard !results.isEmpty else {
            throw FakeSileroVADPredictorError.missingPrediction
        }
        return try results.removeFirst().get()
    }
}
