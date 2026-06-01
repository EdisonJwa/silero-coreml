import CoreML
import Foundation
import SileroCoreML

private let executableName = URL(fileURLWithPath: CommandLine.arguments.first ?? "SileroVADBenchmark").lastPathComponent

private struct BenchmarkOptions {
    var modelPaths: [String] = []
    var chunks = 100
    var loadRepetitions = 5
    var computeUnits = "default"
    var jsonOutput: String?
}

private struct BenchmarkReport: Encodable {
    let benchmark: String
    let sampleRate: Int
    let chunkSize: Int
    let computeUnits: String
    let chunks: Int
    let loadRepetitions: Int
    let results: [ModelBenchmarkResult]
}

private struct ModelBenchmarkResult: Encodable {
    let artifactKind: String
    let modelPath: String
    let computeUnits: String
    let loadTimings: TimingSummary
    let firstPredictionTiming: TimingSummary
    let steadyState: SteadyStateSummary
    let chunks: Int
    let audioSeconds: Double
    let rtf: Double
    let rtfx: Double
    let minProbability: Float
    let maxProbability: Float
}

private struct TimingSummary: Encodable {
    let totalSeconds: Double
    let meanSeconds: Double
    let p50Seconds: Double
    let p95Seconds: Double
    let maxSeconds: Double
    let samples: Int
}

private struct SteadyStateSummary: Encodable {
    let totalSeconds: Double
    let meanSeconds: Double
    let p50Seconds: Double
    let p95Seconds: Double
    let maxSeconds: Double
}

private enum BenchmarkError: Error, LocalizedError {
    case usage(String)
    case invalidComputeUnits(String)
    case invalidPositiveInteger(option: String, value: String)
    case invalidModelPath(String)

    var errorDescription: String? {
        switch self {
        case let .usage(message):
            return message
        case let .invalidComputeUnits(value):
            return "Unsupported compute units '\(value)'. Use default, all, cpuOnly, cpuAndGPU, or cpuAndNeuralEngine."
        case let .invalidPositiveInteger(option, value):
            return "\(option) must be a positive integer, got '\(value)'."
        case let .invalidModelPath(path):
            return "Model path must end in .mlmodel, .mlpackage, or .mlmodelc: \(path)"
        }
    }
}

private func usage() -> String {
    """
    Usage: swift run \(executableName) --model-path path/to/SileroVAD.mlpackage [--model-path path/to/SileroVAD.mlmodelc]

    Options:
      --model-path PATH             CoreML artifact to benchmark. Repeat to compare artifacts.
      --chunks N                    Number of deterministic 512-sample chunks for steady-state timing. Default: 100.
      --load-repetitions N          Number of SileroVAD construction/load timings. Default: 5.
      --compute-units VALUE         default|all|cpuOnly|cpuAndGPU|cpuAndNeuralEngine. Default: default.
      --json-output PATH            Write the JSON report to PATH; stdout stays human-readable summary only.
      --help                        Show this help.
    """
}

private func parseArguments(_ arguments: [String]) throws -> BenchmarkOptions? {
    var options = BenchmarkOptions()
    var index = 1

    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--help", "-h":
            return nil
        case "--model-path":
            options.modelPaths.append(try value(after: argument, in: arguments, index: &index))
        case "--chunks":
            options.chunks = try positiveInteger(option: argument, value: value(after: argument, in: arguments, index: &index))
        case "--load-repetitions":
            options.loadRepetitions = try positiveInteger(option: argument, value: value(after: argument, in: arguments, index: &index))
        case "--compute-units":
            options.computeUnits = try value(after: argument, in: arguments, index: &index)
        case "--json-output":
            options.jsonOutput = try value(after: argument, in: arguments, index: &index)
        default:
            throw BenchmarkError.usage("Unknown argument: \(argument)\n\n\(usage())")
        }
        index += 1
    }

    guard !options.modelPaths.isEmpty else {
        throw BenchmarkError.usage("At least one --model-path is required.\n\n\(usage())")
    }
    for modelPath in options.modelPaths {
        guard ["mlmodel", "mlpackage", "mlmodelc"].contains(URL(fileURLWithPath: modelPath).pathExtension) else {
            throw BenchmarkError.invalidModelPath(modelPath)
        }
    }
    _ = try configuration(for: options.computeUnits)
    return options
}

private func value(after option: String, in arguments: [String], index: inout Int) throws -> String {
    let valueIndex = index + 1
    guard valueIndex < arguments.count else {
        throw BenchmarkError.usage("Missing value for \(option).\n\n\(usage())")
    }
    index = valueIndex
    return arguments[valueIndex]
}

private func positiveInteger(option: String, value: String) throws -> Int {
    guard let parsed = Int(value), parsed > 0 else {
        throw BenchmarkError.invalidPositiveInteger(option: option, value: value)
    }
    return parsed
}

private func configuration(for computeUnits: String) throws -> MLModelConfiguration {
    let configuration = MLModelConfiguration()
    switch computeUnits {
    case "default":
        break
    case "all":
        configuration.computeUnits = .all
    case "cpuOnly":
        configuration.computeUnits = .cpuOnly
    case "cpuAndGPU":
        configuration.computeUnits = .cpuAndGPU
    case "cpuAndNeuralEngine":
        configuration.computeUnits = .cpuAndNeuralEngine
    default:
        throw BenchmarkError.invalidComputeUnits(computeUnits)
    }
    return configuration
}

private func measure(_ work: () throws -> Void) rethrows -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    try work()
    let end = DispatchTime.now().uptimeNanoseconds
    return Double(end - start) / 1_000_000_000
}

private func deterministicChunk(index: Int) -> [Float] {
    (0..<SileroVAD.chunkSize).map { sampleIndex in
        let phase = Float((index * SileroVAD.chunkSize + sampleIndex) % 160) / 160
        return sin(phase * 2 * Float.pi) * 0.01
    }
}

private func summarize(_ values: [Double]) -> TimingSummary {
    let sorted = values.sorted()
    let total = values.reduce(0, +)
    return TimingSummary(
        totalSeconds: total,
        meanSeconds: total / Double(values.count),
        p50Seconds: percentile(sorted, 0.50),
        p95Seconds: percentile(sorted, 0.95),
        maxSeconds: sorted.last ?? 0,
        samples: values.count
    )
}

private func percentile(_ sortedValues: [Double], _ percentile: Double) -> Double {
    guard !sortedValues.isEmpty else { return 0 }
    let index = Int(ceil(percentile * Double(sortedValues.count))) - 1
    return sortedValues[min(max(index, 0), sortedValues.count - 1)]
}

private func benchmark(modelPath: String, options: BenchmarkOptions) throws -> ModelBenchmarkResult {
    let modelURL = URL(fileURLWithPath: modelPath)
    let absolutePath = modelURL.path
    var loadTimings: [Double] = []
    var firstPredictionTimings: [Double] = []
    let firstChunk = deterministicChunk(index: 0)

    for _ in 0..<options.loadRepetitions {
        var vad: SileroVAD?
        loadTimings.append(try measure {
            vad = try SileroVAD(modelURL: modelURL, configuration: configuration(for: options.computeUnits))
        })
        guard let loadedVAD = vad else { continue }
        firstPredictionTimings.append(try measure {
            _ = try loadedVAD.process(firstChunk)
        })
    }

    let steadyVad = try SileroVAD(modelURL: modelURL, configuration: configuration(for: options.computeUnits))
    var predictionTimings: [Double] = []
    var probabilities: [Float] = []
    _ = try steadyVad.process(firstChunk)
    let steadyTotal = try measure {
        for chunkIndex in 0..<options.chunks {
            let chunk = deterministicChunk(index: chunkIndex + 1)
            var probability: Float = 0
            let duration = try measure {
                probability = try steadyVad.process(chunk)
            }
            predictionTimings.append(duration)
            probabilities.append(probability)
        }
    }

    let audioSeconds = Double(options.chunks * SileroVAD.chunkSize) / Double(SileroVAD.sampleRate)
    let steadySummary = summarize(predictionTimings)
    return ModelBenchmarkResult(
        artifactKind: artifactKind(for: modelURL),
        modelPath: absolutePath,
        computeUnits: options.computeUnits,
        loadTimings: summarize(loadTimings),
        firstPredictionTiming: summarize(firstPredictionTimings),
        steadyState: SteadyStateSummary(
            totalSeconds: steadyTotal,
            meanSeconds: steadySummary.meanSeconds,
            p50Seconds: steadySummary.p50Seconds,
            p95Seconds: steadySummary.p95Seconds,
            maxSeconds: steadySummary.maxSeconds
        ),
        chunks: options.chunks,
        audioSeconds: audioSeconds,
        rtf: steadyTotal / audioSeconds,
        rtfx: audioSeconds / steadyTotal,
        minProbability: probabilities.min() ?? 0,
        maxProbability: probabilities.max() ?? 0
    )
}

private func artifactKind(for modelURL: URL) -> String {
    switch modelURL.pathExtension {
    case "mlmodel":
        return ".mlmodel"
    case "mlpackage":
        return ".mlpackage"
    case "mlmodelc":
        return ".mlmodelc"
    default:
        return "unknown"
    }
}

private func printHumanSummary(_ report: BenchmarkReport) {
    for result in report.results {
        print("\(result.artifactKind) \(result.modelPath)")
        print("  load mean: \(result.loadTimings.meanSeconds)s p95: \(result.loadTimings.p95Seconds)s")
        print("  first prediction mean: \(result.firstPredictionTiming.meanSeconds)s")
        print("  steady mean: \(result.steadyState.meanSeconds)s p95: \(result.steadyState.p95Seconds)s rtf: \(result.rtf) rtfx: \(result.rtfx)")
        print("  probabilities min/max: \(result.minProbability)/\(result.maxProbability)")
    }
}

do {
    guard let options = try parseArguments(CommandLine.arguments) else {
        print(usage())
        exit(EXIT_SUCCESS)
    }

    let results = try options.modelPaths.map { try benchmark(modelPath: $0, options: options) }
    let report = BenchmarkReport(
        benchmark: "SileroVADBenchmark",
        sampleRate: SileroVAD.sampleRate,
        chunkSize: SileroVAD.chunkSize,
        computeUnits: options.computeUnits,
        chunks: options.chunks,
        loadRepetitions: options.loadRepetitions,
        results: results
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let jsonData = try encoder.encode(report)

    if let jsonOutput = options.jsonOutput {
        let outputURL = URL(fileURLWithPath: jsonOutput)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try jsonData.write(to: outputURL)
    }

    printHumanSummary(report)
    if options.jsonOutput == nil, let json = String(data: jsonData, encoding: .utf8) {
        print(json)
    }
} catch {
    FileHandle.standardError.write("Error: \(error.localizedDescription)\n".data(using: .utf8)!)
    exit(EX_USAGE)
}
