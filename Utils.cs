using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;
using System;
using System.Linq;

public static class Utils
{
    public static Image<Rgb24> Pad(Image<Rgb24> img, int divable = 32)
    {
        const int threshold = 128;


        int[] dims = new int[2];
        int width = img.Width;
        int height = img.Height; foreach (var dimIndex in new int[] { 0, 1 })
        {
            int size = (dimIndex == 0) ? width : height;
            int div = size / divable;
            int mod = size % divable;
            dims[dimIndex] = divable * (div + (mod > 0 ? 1 : 0));
        }

        // 调整图像大小
        img.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Mode = ResizeMode.Pad,
            Size = new Size(dims[0], dims[1]),
            PadColor = Color.White
        }));

        return img;

        //// Convert image to grayscale
        //var grayImage = img.Clone(ctx => ctx.Grayscale());

        //// Extract pixel data as a 2D array
        //byte[] data = new byte[grayImage.Width * grayImage.Height];
        //for (int y = 0; y < grayImage.Height; y++)
        //{
        //    for (int x = 0; x < grayImage.Width; x++)
        //    {
        //        data[y * grayImage.Width + x] = grayImage[x, y].R;
        //    }
        //}

        //// Normalize the pixel data
        //byte minVal = data.Min();
        //byte maxVal = data.Max();
        //float range = maxVal - minVal;
        //float[] normalized = data.Select(p => (p - minVal) / range * 255).ToArray();

        //// Invert the image if the mean is greater than the threshold
        //if (normalized.Average() > threshold)
        //{
        //    normalized = normalized.Select(p => 255 - p).ToArray();
        //}

        // Find the bounding box of the non-zero pixels
        //int width = grayImage.Width;
        //int height = grayImage.Height;
        //var indices = normalized.Select((value, index) => (value > 0 ? index : -1))
        //    .Where(index => index >= 0)
        //    .Select(index => (x: index % width, y: index / width));

        //int minX = indices.Min(p => p.x);
        //int minY = indices.Min(p => p.y);
        //int maxX = indices.Max(p => p.x);
        //int maxY = indices.Max(p => p.y);

        // Crop the image to the bounding box
        // Find bounding box
        //var gray = new Image<L8>(image.Width, image.Height);
        //gray.ProcessPixelRows(accessor =>
        //{
        //    int idx = 0;
        //    for (int y = 0; y < accessor.Height; y++)
        //    {
        //        var row = accessor.GetRowSpan(y);
        //        for (int x = 0; x < accessor.Width; x++)
        //        {
        //            row[x] = new L8(data[idx++] < threshold ? (byte)0 : (byte)255);
        //        }
        //    }
        //});

        //var bounds = gray.Bounds();
        //var cropped = image.Clone(ctx => ctx.Crop(bounds));

        //// Calculate padded size
        //int paddedWidth = GetNextDivisible(cropped.Width, divisible);
        //int paddedHeight = GetNextDivisible(cropped.Height, divisible);

        //// Pad the image
        //var padded = new Image<L8>(paddedWidth, paddedHeight);
        //padded.Mutate(ctx => ctx.Fill(Color.White));
        //padded.Mutate(ctx => ctx.DrawImage(cropped, new Point(0, 0), 1));

        //return padded;
    }
}
