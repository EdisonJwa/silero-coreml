import Foundation

public struct SileroVADSegmenterConfiguration: Equatable, Sendable {
    public var entryThreshold: Float
    public var exitThreshold: Float
    public var minSpeechDuration: TimeInterval
    public var minSilenceDuration: TimeInterval
    public var speechPadding: TimeInterval

    public init(
        entryThreshold: Float = SileroVADRunner.defaultThreshold,
        exitThreshold: Float = SileroVADRunner.defaultThreshold,
        minSpeechDuration: TimeInterval = 0,
        minSilenceDuration: TimeInterval = 0,
        speechPadding: TimeInterval = 0
    ) {
        precondition(entryThreshold.isFinite, "entryThreshold must be finite")
        precondition(exitThreshold.isFinite, "exitThreshold must be finite")
        precondition(entryThreshold >= exitThreshold, "entryThreshold must be greater than or equal to exitThreshold")
        precondition(minSpeechDuration.isFinite, "minSpeechDuration must be finite")
        precondition(minSpeechDuration >= 0, "minSpeechDuration must be non-negative")
        precondition(minSilenceDuration.isFinite, "minSilenceDuration must be finite")
        precondition(minSilenceDuration >= 0, "minSilenceDuration must be non-negative")
        precondition(speechPadding.isFinite, "speechPadding must be finite")
        precondition(speechPadding >= 0, "speechPadding must be non-negative")

        self.entryThreshold = entryThreshold
        self.exitThreshold = exitThreshold
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.speechPadding = speechPadding
    }
}

public struct SileroVADSegment: Equatable, Sendable {
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let startFrameIndex: Int
    public let endFrameIndex: Int

    public var duration: TimeInterval {
        endTime - startTime
    }

    public init(startFrameIndex: Int, endFrameIndex: Int, frameDuration: TimeInterval) {
        self.startFrameIndex = startFrameIndex
        self.endFrameIndex = endFrameIndex
        startTime = TimeInterval(startFrameIndex) * frameDuration
        endTime = TimeInterval(endFrameIndex) * frameDuration
    }
}

public struct SileroVADSegmenter: Sendable {
    public let configuration: SileroVADSegmenterConfiguration
    public let sampleRate: Int
    public let chunkSize: Int
    public let frameDuration: TimeInterval

    private let minSpeechFrames: Int
    private let minSilenceFrames: Int
    private let paddingFrames: Int

    private var nextFrameIndex = 0
    private var candidateStartFrameIndex: Int?
    private var openStartFrameIndex: Int?
    private var firstSilenceFrameIndex: Int?

    public init(
        configuration: SileroVADSegmenterConfiguration = SileroVADSegmenterConfiguration(),
        sampleRate: Int = SileroVADRunner.sampleRate,
        chunkSize: Int = SileroVADRunner.chunkSize
    ) {
        precondition(sampleRate > 0, "sampleRate must be greater than zero")
        precondition(chunkSize > 0, "chunkSize must be greater than zero")

        self.configuration = configuration
        self.sampleRate = sampleRate
        self.chunkSize = chunkSize
        frameDuration = TimeInterval(chunkSize) / TimeInterval(sampleRate)
        minSpeechFrames = Self.frameCount(for: configuration.minSpeechDuration, frameDuration: frameDuration)
        minSilenceFrames = Self.frameCount(for: configuration.minSilenceDuration, frameDuration: frameDuration)
        paddingFrames = Self.frameCount(for: configuration.speechPadding, frameDuration: frameDuration)
    }

    public mutating func process(probability: Float) -> [SileroVADSegment] {
        defer { nextFrameIndex += 1 }

        if let openStartFrameIndex {
            return processOpenSegment(probability: probability, openStartFrameIndex: openStartFrameIndex)
        }

        return processCandidate(probability: probability)
    }

    public mutating func process(probabilities: [Float]) -> [SileroVADSegment] {
        probabilities.flatMap { process(probability: $0) }
    }

    public mutating func finalize() -> [SileroVADSegment] {
        defer {
            candidateStartFrameIndex = nil
            openStartFrameIndex = nil
            firstSilenceFrameIndex = nil
        }

        if let openStartFrameIndex {
            return [makeSegment(
                startFrameIndex: openStartFrameIndex,
                endFrameIndex: nextFrameIndex,
                processedFrameCount: nextFrameIndex
            )]
        }

        guard let candidateStartFrameIndex,
              nextFrameIndex - candidateStartFrameIndex >= minSpeechFrames else {
            return []
        }

        return [makeSegment(
            startFrameIndex: candidateStartFrameIndex,
            endFrameIndex: nextFrameIndex,
            processedFrameCount: nextFrameIndex
        )]
    }

    public mutating func reset() {
        nextFrameIndex = 0
        candidateStartFrameIndex = nil
        openStartFrameIndex = nil
        firstSilenceFrameIndex = nil
    }

    private mutating func processCandidate(probability: Float) -> [SileroVADSegment] {
        if probability >= configuration.entryThreshold {
            if candidateStartFrameIndex == nil {
                candidateStartFrameIndex = nextFrameIndex
            }
            qualifyCandidateIfNeeded()
            return []
        }

        if probability >= configuration.exitThreshold, candidateStartFrameIndex != nil {
            qualifyCandidateIfNeeded()
            return []
        }

        candidateStartFrameIndex = nil
        return []
    }

    private mutating func processOpenSegment(
        probability: Float,
        openStartFrameIndex: Int
    ) -> [SileroVADSegment] {
        guard probability < configuration.exitThreshold else {
            firstSilenceFrameIndex = nil
            return []
        }

        if firstSilenceFrameIndex == nil {
            firstSilenceFrameIndex = nextFrameIndex
        }

        guard let firstSilenceFrameIndex,
              nextFrameIndex - firstSilenceFrameIndex + 1 >= minSilenceFrames else {
            return []
        }

        let segment = makeSegment(
            startFrameIndex: openStartFrameIndex,
            endFrameIndex: firstSilenceFrameIndex,
            processedFrameCount: nextFrameIndex + 1
        )
        self.openStartFrameIndex = nil
        self.firstSilenceFrameIndex = nil
        candidateStartFrameIndex = nil
        return [segment]
    }

    private mutating func qualifyCandidateIfNeeded() {
        guard let candidateStartFrameIndex,
              nextFrameIndex - candidateStartFrameIndex + 1 >= minSpeechFrames else {
            return
        }

        openStartFrameIndex = candidateStartFrameIndex
        self.candidateStartFrameIndex = nil
    }

    private func makeSegment(
        startFrameIndex: Int,
        endFrameIndex: Int,
        processedFrameCount: Int
    ) -> SileroVADSegment {
        let paddedStartFrameIndex = max(0, startFrameIndex - paddingFrames)
        let paddedEndFrameIndex = min(processedFrameCount, endFrameIndex + paddingFrames)
        return SileroVADSegment(
            startFrameIndex: paddedStartFrameIndex,
            endFrameIndex: paddedEndFrameIndex,
            frameDuration: frameDuration
        )
    }

    private static func frameCount(for duration: TimeInterval, frameDuration: TimeInterval) -> Int {
        guard duration > 0 else {
            return 0
        }

        return max(1, Int(ceil(duration / frameDuration)))
    }
}
