using System;
using System.Collections.Generic;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    public class CustomMetricResult
    {
        public float sum_metric { get; set; }
        public int num_inst { get; set; }
    }
    public delegate CustomMetricResult CustomMetricEval(SingleNArray labels, SingleNArray preds);
    public class CustomMetric : EvalMetric
    {
        private readonly CustomMetricEval _feval;

        public CustomMetric
            (CustomMetricEval eval, string name = null, int num = 1) : base(GetName(eval, name), num)
        {
            _feval = eval;
        }

        private static string GetName(CustomMetricEval eval, string name)
        {
            return !string.IsNullOrWhiteSpace(name) ? name : $"Custom({eval.Method.Name})";
        }

        public override void Update(List<NDArray> labels, List<NDArray> preds)
        {
            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].as_numerics();
                var pred = preds[i].as_numerics();

                if (pred.shape[1] == 2)
                {
                    //TODO pred = pred[:, 1]
                }
                var reval = this._feval(label, pred);
                this.sum_metric[0] += reval.sum_metric;
                this.num_inst[0] += reval.num_inst;
            }
        }
    }
}