namespace NNSandbox.NN
{
    internal abstract class ActivationFunction
    {
        public abstract float Calculate(float x);

        public abstract float CalculateDerivative(float x);
    }
}
