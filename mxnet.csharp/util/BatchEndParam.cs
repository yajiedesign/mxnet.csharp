using System.Globalization;
using mxnet.csharp.metric;

namespace mxnet.csharp.util
{
    public class BatchEndParam
    {
        public int epoch { get; set; }
        public int nbatch { get; set; }
        public EvalMetric eval_metric { get; set; }
        public CultureInfo locals { get; set; }

        public BatchEndParam(int epoch, int nbatch, EvalMetric eval_metric, CultureInfo locals)
        {
            this.epoch = epoch;
            this.nbatch = nbatch;
            this.eval_metric = eval_metric;
            this.locals = locals;
        }
    }
}