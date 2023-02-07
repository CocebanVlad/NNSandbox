namespace NNSandbox
{
    internal static class Utils
    {
        public static void PrintArray<T>(T[] array) =>
            Console.WriteLine($"[ {string.Join(", ", array)} ]");

        public static void PrintArray<T>(T[][] array)
        {
            Console.WriteLine("[");

            foreach (var subarray in array)
            {
                Console.WriteLine($"[ {string.Join(", ", subarray)} ]");
            }

            Console.WriteLine("]");
        }
    }
}
