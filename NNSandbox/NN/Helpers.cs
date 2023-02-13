namespace NNSandbox.NN
{
    internal static class Helpers
    {
        public static T[] CreateArray<T>(int size, Func<T> factory) => Enumerable.Range(0, size).Select(_ => factory()).ToArray();

        public static T[] CreateArray<T>(int size, T value) => CreateArray(size, () => value);

        public static float GetRandomFloat(Random r, float min, float max) => min + (max - min) * r.NextSingle();
    }
}
