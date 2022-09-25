using MathNet.Numerics.LinearAlgebra;

namespace lijohnttle.Learning.NeuN.Core.Internal
{
    internal interface IQueryProcessor
    {
        Vector<double>[] Query(
            Vector<double> inputs,
            Matrix<double>[] weightsLayers,
            Func<double, double> activationFunction);
    }
}
