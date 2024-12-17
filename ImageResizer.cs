using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static System.Net.Mime.MediaTypeNames;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

public class ImageResizer
{
    static readonly string preprocessModelPath = @"D:\project\RapidLaTeXOCR\rapid_latex_ocr\models\image_resizer.onnx"; // Replace with the actual path


    public static Image<Rgb24> LoopImageResizer(Image<Rgb24> img)
    {
        // Step 1: Padding and resizing input image
        Image<Rgb24> padImg = Utils.Pad(img);
        //padImg.Save("tem.png");
        Image<Rgb24> inputImage = MinMaxSize(padImg);

        float r = 1.0f;
        int w = inputImage.Width;
        int h = inputImage.Height;
        Image<Rgb24> finalImg = null;

        for (int i = 0; i < 10; i++)
        {
            h = (int)(h * r);
            (finalImg, padImg) = PreProcess(inputImage, r, w, h);

            // Simulate image resizer model output (replace with actual model logic)
            float[] resizerRes = ImageResizerModel(finalImg);
            int argmaxIdx = ArgMax(resizerRes);

            w = (argmaxIdx + 1) * 32;
            if (w == padImg.Width)
                break;

            r = (float)w / padImg.Width;
        }

        return finalImg;
    }

    private static float[] ImageResizerModel(Image<Rgb24> image)
    {
        // Load and preprocess image using the ONNX preprocessing model
        using var preprocessSession = new InferenceSession(preprocessModelPath);
        string inputName = preprocessSession.InputMetadata.Keys.First();

        var mean = new[] { 0.7931f, 0.7931f, 0.7931f };
        var stddev = new[] { 0.1738f, 0.1738f, 0.1738f };
        DenseTensor<float> processedImage = new(new[] { 1, 1, image.Height, image.Width });
        image.ProcessPixelRows(accessor =>
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
        var preprocessInputContainer = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, processedImage)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> preprocessResults = preprocessSession.Run(preprocessInputContainer);

        string outputName = preprocessSession.OutputMetadata.Keys.First();
        return preprocessResults.First(r => r.Name == outputName).AsTensor<float>().ToArray();
    }

    private static int ArgMax(float[] array)
    {
        return array.ToList().IndexOf(array.Max());
    }

    public static Image<Rgb24> MinMaxSize(Image<Rgb24> img)
    {
        return img;
        // Resize image to specific size range (dummy example: 256x256)
        //int newWidth = Math.Min(256, img.Width);
        //int newHeight = Math.Min(256, img.Height);
        //return img.Clone(context => context.Resize(newWidth, newHeight));
    }


    public static (Image<Rgb24>, Image<Rgb24>) PreProcess(Image<Rgb24> inputImage, float ratio, int w, int h)
    {
        IResampler resampler;

        // 根据缩放比例选择不同的插值算法
        if (ratio > 1)
        {
            resampler = KnownResamplers.Bicubic; // 放大时使用Bilinear插值 ?
        }
        else
        {
            resampler = KnownResamplers.Lanczos3; // 缩小时使用Lanczos插值
        }

        // 调整图像大小
        inputImage.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Mode = ResizeMode.Stretch,
            Size = new Size(w, h),
            Sampler = resampler
        }));

        // 填充操作 (模拟pad)
        Image<Rgb24> paddedImage = Utils.Pad(inputImage);

        // 转换为RGB格式
        //paddedImage.Mutate(ctx => ctx.BackgroundColor(SixLabors.ImageSharp.Color.Black));
        // 转换为灰度图并归一化
        //var grayImage = ToGrayScale(paddedImage);
        //var normalizedImage = Normalize(grayImage);

        return (inputImage, paddedImage);
    }
}

