using NNSandbox;

Console.WriteLine("Hello, World!");

var data = new List<NeuralNetwork.TrainingDataSetEntry>();
var bit = new float[] { 0.1f, 1.0f };

for (var i = 0; i < 1; i++)
{
    data.Add(new(new float[] { bit[0], bit[0], bit[0] }, new float[] { bit[0] }));
    data.Add(new(new float[] { bit[0], bit[1], bit[0] }, new float[] { bit[1] }));
    data.Add(new(new float[] { bit[1], bit[0], bit[0] }, new float[] { bit[1] }));
    data.Add(new(new float[] { bit[1], bit[1], bit[0] }, new float[] { bit[1] }));
    data.Add(new(new float[] { bit[0], bit[0], bit[1] }, new float[] { bit[0] }));
    data.Add(new(new float[] { bit[0], bit[1], bit[1] }, new float[] { bit[0] }));
    data.Add(new(new float[] { bit[1], bit[0], bit[1] }, new float[] { bit[0] }));
    data.Add(new(new float[] { bit[1], bit[1], bit[1] }, new float[] { bit[1] }));
}

var nn = new NeuralNetwork(new[] { 3, 6, 9, 6, 1 });

for (var i = 0; i < 10000; i++)
{
    nn.Train(data);

    if (i % 100 != 0)
    {
        continue;
    }

    Console.WriteLine("Epoch " + i);

    Console.WriteLine("0 or 0");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[0], bit[0] })[^1]);

    Console.WriteLine("0 or 1");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[1], bit[0] })[^1]);

    //Console.WriteLine("1 or 1");
    //Utils.PrintArray(nn.Test(new float[] { bit[1], bit[1], bit[0] }));

    //Console.WriteLine("0 and 0");
    //Utils.PrintArray(nn.Test(new float[] { bit[0], bit[0], bit[1] }));

    //Console.WriteLine("0 and 1");
    //Utils.PrintArray(nn.Test(new float[] { bit[0], bit[1], bit[1] }));

    //Console.WriteLine("1 and 1");
    //Utils.PrintArray(nn.Test(new float[] { bit[1], bit[1], bit[1] }));

    Console.WriteLine(new string('-', 10));
}

Console.WriteLine();