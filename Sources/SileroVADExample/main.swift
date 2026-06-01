import Foundation
import SileroCoreML

let arguments = CommandLine.arguments
let vad: SileroVAD

switch arguments.count {
case 1:
    vad = try SileroVAD()
case 2:
    vad = try SileroVAD(modelURL: URL(fileURLWithPath: arguments[1]))
default:
    let executableName = URL(fileURLWithPath: arguments.first ?? "SileroVADExample").lastPathComponent
    FileHandle.standardError.write(
        "Usage: swift run \(executableName) [path/to/SileroVAD.mlpackage]\n".data(using: .utf8)!
    )
    exit(EX_USAGE)
}

let silentChunk = Array(repeating: Float(0), count: SileroVAD.chunkSize)
let probability = try vad.process(silentChunk)

print("Silent chunk speech probability: \(probability)")
