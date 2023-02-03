// See https://aka.ms/new-console-template for more information

using NNSandbox;

Console.WriteLine("Hello, World!");

var nn = new NN(new[] { 2, 3, 2 });

for (var i = 0; i < 2; i++)
{
    nn.Train(new float[] { 1, 1 }, new float[] { 1, 0 });
    nn.Train(new float[] { 0, 1 }, new float[] { 0, 1 });
    nn.Train(new float[] { 1, 0 }, new float[] { 0, 1 });
    nn.Train(new float[] { 0, 0 }, new float[] { 0, 1 });
}

var result = nn.Test(new float[] { 1, 1 });

Console.WriteLine();