using MathNet.Numerics.LinearAlgebra;

namespace lijohnttle.Learning.NeuN.Core.Internal
{
    internal interface INetworkTrainer
    {
        Matrix<double>[] Train(
            Vector<double> inputs,
            Vector<double> targets,
            Matrix<double>[] weightsLayers,
            double learningRate,
            Func<double, double> activationFunction,
            IQueryProcessor queryProcessor);
    }
}
