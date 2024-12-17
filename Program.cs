// Ensure you have the ONNX models for the preprocessing, encoder, and decoder of LaTeX-OCR and ONNX Runtime installed.
// You can install the ONNX Runtime NuGet package using:
// dotnet add package Microsoft.ML.OnnxRuntime

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using TestLatex;

class Program
{
    static void Main(string[] args)
    {
        string preprocessModelPath = @"D:\project\RapidLaTeXOCR\rapid_latex_ocr\models\image_resizer.onnx"; // Replace with the actual path
        string encoderModelPath = @"D:\project\RapidLaTeXOCR\rapid_latex_ocr\models\encoder.onnx"; // Replace with the actual path
        string decoderModelPath = @"D:\project\RapidLaTeXOCR\rapid_latex_ocr\models\decoder.onnx"; // Replace with the actual path
        string imagePath = "6.png"; // Replace with the actual image path

        using var image = Image.Load<Rgb24>(imagePath);
        var preprocessedInput = ImageResizer.LoopImageResizer(image);
        //preprocessedInput.Save("tem.png");

        // Run the encoder model
        using var encoderSession = new InferenceSession(encoderModelPath);
        var encoderInputMeta = encoderSession.InputMetadata;
        string encoderInputName = encoderInputMeta.Keys.First();

        var mean = new[] { 0.7931f, 0.7931f, 0.7931f };
        var stddev = new[] { 0.1738f, 0.1738f, 0.1738f };
        DenseTensor<float> processedImage = new(new[] { 1, 1, preprocessedInput.Height, preprocessedInput.Width });
        preprocessedInput.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    processedImage[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                }
            }
        });
        var encoderInputContainer = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(encoderInputName, processedImage)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = encoderSession.Run(encoderInputContainer);

        // Extract encoder output
        string encoderOutputName = encoderSession.OutputMetadata.Keys.First();
        var encoderOutputTensor = encoderResults.First(r => r.Name == encoderOutputName).AsTensor<float>();

        // Prepare input for the decoder model
        //var decoderInputTensor  = PrepareDecoderInput(encoderOutputTensor);

        var decoderOutputTokens = TokenGenerator.Generate(encoderOutputTensor);

        // Convert tokens to LaTeX code
        string latexCode = DecodeTokens(decoderOutputTokens);

        Console.WriteLine("Generated LaTeX Code:");
    }


    static List<NamedOnnxValue> GetImage(string preprocessModelPath, string imagePath)
    {
        // Load and preprocess image using the ONNX preprocessing model
        using var preprocessSession = new InferenceSession(preprocessModelPath);

        var imageTensor = LoadImageAsTensor(imagePath);
        string inputName = preprocessSession.InputMetadata.Keys.First();
        var preprocessInputContainer = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, imageTensor)
        };
        return preprocessInputContainer;
    }



    public static DenseTensor<float> LoadImageAsTensor(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        image.Mutate(x =>
        {
            x.Resize(new ResizeOptions
            {
                Size = new Size(224, 224),
                Mode = ResizeMode.Crop
            });

            x.Grayscale();
        });


        DenseTensor<float> processedImage = new(new[] { 1, 1, image.Height, image.Width });
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    processedImage[0, 0, y, x] = pixelSpan[x].R / 255f;

                }
            }
        });

        return processedImage;
    }



    static DenseTensor<float> PrepareDecoderInput(Tensor<float> encoderOutput)
    {
        // Assuming the decoder requires sequence data from the encoder
        // Prepare the initial states and input tensors
        int sequenceLength = encoderOutput.Dimensions[1];
        int featureSize = encoderOutput.Dimensions[2];

        var decoderInput = new DenseTensor<float>(new[] { 1, sequenceLength, featureSize });
        var decoderInitialStates = new List<DenseTensor<float>>();

        for (int i = 0; i < sequenceLength; i++)
        {
            var stateTensor = new DenseTensor<float>(new[] { 1, featureSize });
            for (int j = 0; j < featureSize; j++)
            {
                stateTensor[0, j] = encoderOutput[0, i, j];
            }
            decoderInitialStates.Add(stateTensor);
        }

        return decoderInput;
    }

    static int PostprocessOutput(Tensor<float> outputTensor)
    {
        // Decode the output tensor into a token ID (integer)
        return outputTensor.ToArray().Select((value, index) => (value, index)).OrderByDescending(x => x.value).First().index;
    }

    static string DecodeTokens(List<long> tokens)
    {
        var vocabulary = LoadVocabulary("tokenizer.json");
        var result = string.Join("", tokens.Select(token => vocabulary[(int)token]));

        result = result.Replace("Ġ","").Replace("臓", " ").Replace("[EOS]", "").Replace("[BOS]", "").Replace("[PAD]", "").Trim();
        return result;
    }

    static Dictionary<int, string> LoadVocabulary(string vocabPath)
    {
        // Load vocabulary from a JSON file
        var vocabJson = File.ReadAllText(vocabPath);
        var bPE = System.Text.Json.JsonSerializer.Deserialize<BPE>(vocabJson);
        bPE.model.ReVocab = bPE.model.vocab.ToDictionary(pair => pair.Value, pair => pair.Key);
        return bPE.model.ReVocab;
    }
}
