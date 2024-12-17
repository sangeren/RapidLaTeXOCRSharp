using System;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class TokenGenerator
{
    private static readonly int _maxSeqLen = 512;
    private static readonly int startTokens = 1;
    private static readonly int endTokens = 2;
    static string decoderModelPath = @"D:\project\RapidLaTeXOCR\rapid_latex_ocr\models\decoder.onnx"; // Replace with the actual path



    public static List<Int64> Generate(Tensor<float> context, float temperature = 1.0f, float filterThres = 0.9f)
    {
        using var decoderSession = new InferenceSession(decoderModelPath);

        var outTokens = startTokens;

        var x = new List<Int64> { startTokens };
        var mask = new List<bool> { true };

        for (int i = 0; i < _maxSeqLen; i++)
        {
            var inputTensor = new DenseTensor<Int64>(x.ToArray(), new int[] { 1, x.Count });
            var maskTensor = new DenseTensor<bool>(mask.ToArray(), new int[] { 1, x.Count });


            var decoderInputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("x", inputTensor),
                NamedOnnxValue.CreateFromTensor("mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("context", context),
            };

            var ortOutputs = decoderSession.Run(decoderInputContainer);

            var ortOuts = ortOutputs.First().AsEnumerable<float>().ToArray();

            // Extract logits and filter
            var logits = ortOuts.Skip(i * 8000).ToArray();
            var filteredLogits = TopKFilter(logits, 1, logits.Length, filterThres);

            // Softmax normalization
            var probs = Softmax(filteredLogits.Select(x => x / temperature).ToArray());

            // Sampling step (multinomial)
            var sample = MultinomialSample(probs);

            x.Add(sample);
            mask.Add(true);

            if (endTokens == sample)
            {
                break;
            }
        }

        return x;
    }



    public static float[] TopKFilter(float[] logits, int numRows, int numCols, float thres = 0.9f)
    {
        int k = (int)((1 - thres) * numCols);
        var result = NpTopK(logits, numRows, numCols, k);
        float[] probs = new float[logits.Length];

        // Initialize probs with negative infinity
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] = float.NegativeInfinity;
        }

        // Put top-k values at corresponding positions
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < k; j++)
            {
                int index = i * numCols + result.Item2[i, j]; // Get the flattened index
                probs[index] = result.Item1[i, j];
            }
        }

        return probs;
    }

    private static (float[,] values, int[,] indices) NpTopK(float[] logits, int numRows, int numCols, int k)
    {
        float[,] values = new float[numRows, k];
        int[,] indices = new int[numRows, k];

        for (int i = 0; i < numRows; i++)
        {
            // Extract the row from the flattened logits
            var row = Enumerable.Range(0, numCols)
                                .Select(j => new { Value = logits[i * numCols + j], Index = j })
                                .OrderByDescending(x => x.Value)
                                .Take(k)
                                .ToArray();

            for (int j = 0; j < k; j++)
            {
                values[i, j] = row[j].Value;
                indices[i, j] = row[j].Index;
            }
        }

        return (values, indices);
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expLogits = logits.Select(x => MathF.Exp(x - maxLogit)).ToArray();
        var sumExp = expLogits.Sum();
        return expLogits.Select(x => x / sumExp).ToArray();
    }

    private static int MultinomialSample(float[] probabilities)
    {
        var rnd = new Random();
        var cumulative = probabilities.Scan((acc, x) => acc + x, 0f).ToArray();
        var randValue = (float)rnd.NextDouble();

        for (int i = 0; i < cumulative.Length; i++)
        {
            if (randValue < cumulative[i])
                return i;
        }

        return cumulative.Length - 1;
    }



    private static double[,] Softmax(double[,] logits, double temperature)
    {
        int rows = logits.GetLength(0);
        int cols = logits.GetLength(1);
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = Math.Exp(logits[i, j] / temperature);
                sum += result[i, j];
            }
            for (int j = 0; j < cols; j++)
            {
                result[i, j] /= sum;
            }
        }
        return result;
    }

    private static int[,] Concatenate(int[,] tokens, int[] newTokens)
    {
        int rows = tokens.GetLength(0);
        int cols = tokens.GetLength(1);
        var result = new int[rows, cols + 1];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = tokens[i, j];
            }
            result[i, cols] = newTokens[i];
        }
        return result;
    }





    private static int[] RemoveInitialTokens(int[,] tokens, int start)
    {
        int rows = tokens.GetLength(0);
        int cols = tokens.GetLength(1) - start;
        var result = new int[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = start; j < tokens.GetLength(1); j++)
            {
                result[(i * cols) + (j - start)] = tokens[i, j];
            }
        }
        return result;
    }

    private static int[] MultinomialSample(double[,] probabilities)
    {
        var rand = new Random();
        int rows = probabilities.GetLength(0);
        var samples = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            double cumulative = 0;
            double target = rand.NextDouble();
            for (int j = 0; j < probabilities.GetLength(1); j++)
            {
                cumulative += probabilities[i, j];
                if (target < cumulative)
                {
                    samples[i] = j;
                    break;
                }
            }
        }
        return samples;
    }

    private static double[,] ExtractLogits(object ortOutputs)
    {
        // Replace with ONNX or equivalent extraction logic
        return new double[1, 1];
    }
}

public static class LinqExtensions
{
    public static IEnumerable<float> Scan(this IEnumerable<float> source, Func<float, float, float> func, float seed)
    {
        float accumulator = seed;
        foreach (var item in source)
        {
            accumulator = func(accumulator, item);
            yield return accumulator;
        }
    }
}
