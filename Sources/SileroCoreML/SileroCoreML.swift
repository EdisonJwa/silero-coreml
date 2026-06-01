public enum VadBackend {
    public static let SileroCoreMl = "VadBackend::SileroCoreMl"
}

public enum SileroCoreML {
    public static let backendIdentifier = VadBackend.SileroCoreMl

    public static let sampleRate = 16_000
    public static let chunkSize = 512
    public static let contextSize = 64
    public static let modelInputSize = 576
    public static let stateShape = [2, 1, 128]
    public static let stateSize = 2 * 1 * 128

    public static let inputName = "input"
    public static let stateName = "state_in"
    public static let outputName = "output"
    public static let nextStateName = "stateN"

    public struct StreamingState: Equatable {
        public private(set) var context: [Float]
        public private(set) var state: [Float]

        public init() {
            context = Array(repeating: 0, count: SileroCoreML.contextSize)
            state = Array(repeating: 0, count: SileroCoreML.stateSize)
        }

        public mutating func reset() {
            context = Array(repeating: 0, count: SileroCoreML.contextSize)
            state = Array(repeating: 0, count: SileroCoreML.stateSize)
        }

        @discardableResult
        public mutating func makeModelInput(for chunk: [Float]) throws -> [Float] {
            guard chunk.count == SileroCoreML.chunkSize else {
                throw StreamingStateError.invalidChunkSize(
                    expected: SileroCoreML.chunkSize,
                    actual: chunk.count
                )
            }

            let modelInput = context + chunk
            context = Array(modelInput.suffix(SileroCoreML.contextSize))
            return modelInput
        }

        public mutating func updateState(with nextState: [Float]) throws {
            guard nextState.count == SileroCoreML.stateSize else {
                throw StreamingStateError.invalidStateSize(
                    expected: SileroCoreML.stateSize,
                    actual: nextState.count
                )
            }

            state = nextState
        }
    }

    public enum StreamingStateError: Error, Equatable {
        case invalidChunkSize(expected: Int, actual: Int)
        case invalidStateSize(expected: Int, actual: Int)

        public var errorDescription: String? {
            switch self {
            case let .invalidChunkSize(expected, actual):
                return "Expected chunk of \(expected) samples but received \(actual)."
            case let .invalidStateSize(expected, actual):
                return "Expected flattened state of \(expected) samples but received \(actual)."
            }
        }
    }
}
