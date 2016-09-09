using System.Globalization;
using mxnet.csharp.metric;

namespace mxnet.csharp.util
{
    public class BatchEndParam
    {
        public int Epoch { get; set; }
        public int Nbatch { get; set; }
        public EvalMetric EvalMetric { get; set; }
        public CultureInfo Locals { get; set; }

        public BatchEndParam(int epoch, int nbatch, EvalMetric evalMetric, CultureInfo locals)
        {
            this.Epoch = epoch;
            this.Nbatch = nbatch;
            this.EvalMetric = evalMetric;
            this.Locals = locals;
        }
    }
}