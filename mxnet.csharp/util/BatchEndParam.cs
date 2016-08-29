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

        public BatchEndParam(int epoch, int nbatch, EvalMetric eval_metric, CultureInfo locals)
        {
            Epoch = epoch;
            Nbatch = nbatch;
            EvalMetric = eval_metric;
            Locals = locals;
        }
    }
}