import XCTest
@testable import SileroCoreML

final class SileroVADSegmenterTests: XCTestCase {
    func testConfigurationDefaultsAndFrameConstantsAreSensible() {
        let configuration = SileroVADSegmenterConfiguration()
        let segmenter = SileroVADSegmenter(configuration: configuration)

        XCTAssertEqual(configuration.entryThreshold, SileroVAD.defaultThreshold)
        XCTAssertEqual(configuration.exitThreshold, SileroVAD.defaultThreshold)
        XCTAssertEqual(configuration.minSpeechDuration, 0)
        XCTAssertEqual(configuration.minSilenceDuration, 0)
        XCTAssertEqual(configuration.speechPadding, 0)
        XCTAssertEqual(segmenter.sampleRate, 16_000)
        XCTAssertEqual(segmenter.chunkSize, 512)
        XCTAssertEqual(segmenter.frameDuration, 0.032, accuracy: 0.000_001)
    }

    func testShortNoiseIsIgnoredIncludingAfterFinalize() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 3, minSilenceFrames: 1))

        XCTAssertEqual(segmenter.process(probabilities: [0.6, 0.6, 0.1]), [])
        XCTAssertEqual(segmenter.finalize(), [])

        var openNoiseSegmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 3, minSilenceFrames: 1))
        XCTAssertEqual(openNoiseSegmenter.process(probabilities: [0.6, 0.6]), [])
        XCTAssertEqual(openNoiseSegmenter.finalize(), [])
    }

    func testSustainedSpeechOpensAndClosesIntoSegment() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 2))

        let segments = segmenter.process(probabilities: [0.6, 0.6, 0.6, 0.1, 0.1])

        XCTAssertEqual(segments, [segment(start: 0, end: 3)])
    }

    func testShortSilenceMergesOneSegment() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 2))

        let segments = segmenter.process(probabilities: [0.6, 0.6, 0.1, 0.6, 0.6, 0.1, 0.1])

        XCTAssertEqual(segments, [segment(start: 0, end: 5)])
    }

    func testLongSilenceClosesSegment() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 3))

        XCTAssertEqual(segmenter.process(probabilities: [0.6, 0.6, 0.1, 0.1]), [])
        XCTAssertEqual(segmenter.process(probability: 0.1), [segment(start: 0, end: 2)])
    }

    func testZeroMinSilenceDurationClosesOnFirstBelowExitFrame() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 0))

        XCTAssertEqual(segmenter.process(probabilities: [0.6, 0.6]), [])
        XCTAssertEqual(segmenter.process(probability: 0.1), [segment(start: 0, end: 2)])
    }

    func testEntryThresholdEqualityCanStartAndQualifySpeech() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 1))

        let segments = segmenter.process(probabilities: [0.5, 0.5, 0.1])

        XCTAssertEqual(segments, [segment(start: 0, end: 2)])
    }

    func testExitThresholdEqualityKeepsOpenSegmentAliveUntilBelowExit() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 1, minSilenceFrames: 1))

        XCTAssertEqual(segmenter.process(probability: 0.6), [])
        XCTAssertEqual(segmenter.process(probability: 0.3), [])
        XCTAssertEqual(segmenter.process(probability: 0.29), [segment(start: 0, end: 2)])
    }

    func testPaddingClampsStartToZeroAndExtendsEndToProcessedFrames() {
        var segmenter = SileroVADSegmenter(
            configuration: configuration(minSpeechFrames: 1, minSilenceFrames: 2, paddingFrames: 2)
        )

        let segments = segmenter.process(probabilities: [0.6, 0.1, 0.1])

        XCTAssertEqual(segments, [segment(start: 0, end: 3)])
    }

    func testFinalizeClosesOpenQualifiedSegment() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 2, paddingFrames: 1))

        XCTAssertEqual(segmenter.process(probabilities: [0.1, 0.6, 0.6, 0.6]), [])

        XCTAssertEqual(segmenter.finalize(), [segment(start: 0, end: 4)])
        XCTAssertEqual(segmenter.finalize(), [])
    }

    func testResetClearsStateAndFrameIndexing() {
        var segmenter = SileroVADSegmenter(configuration: configuration(minSpeechFrames: 2, minSilenceFrames: 1))

        XCTAssertEqual(segmenter.process(probabilities: [0.1, 0.6]), [])
        segmenter.reset()

        let segments = segmenter.process(probabilities: [0.6, 0.6, 0.1])

        XCTAssertEqual(segments, [segment(start: 0, end: 2)])
    }

    private func configuration(
        minSpeechFrames: Int,
        minSilenceFrames: Int,
        paddingFrames: Int = 0
    ) -> SileroVADSegmenterConfiguration {
        let frameDuration = TimeInterval(SileroVAD.chunkSize) / TimeInterval(SileroVAD.sampleRate)
        return SileroVADSegmenterConfiguration(
            entryThreshold: 0.5,
            exitThreshold: 0.3,
            minSpeechDuration: frameDuration * TimeInterval(minSpeechFrames),
            minSilenceDuration: frameDuration * TimeInterval(minSilenceFrames),
            speechPadding: frameDuration * TimeInterval(paddingFrames)
        )
    }

    private func segment(start: Int, end: Int) -> SileroVADSegment {
        let frameDuration = TimeInterval(SileroVAD.chunkSize) / TimeInterval(SileroVAD.sampleRate)
        return SileroVADSegment(startFrameIndex: start, endFrameIndex: end, frameDuration: frameDuration)
    }
}
