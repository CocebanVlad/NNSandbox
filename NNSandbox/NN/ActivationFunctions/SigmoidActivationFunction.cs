namespace NNSandbox.NN.ActivationFunctions
{
    internal class SigmoidActivationFunction : ActivationFunction
    {
        public override float Calculate(float x) => 1f / (1f + (float)Math.Exp(-x));

        public override float CalculateDerivative(float x) => x * (1 - x);
    }
}
