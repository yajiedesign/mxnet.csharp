using System;
using System.Collections.Generic;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    public class CustomMetricResult
    {
        public float SumMetric { get; set; }
        public int NumInst { get; set; }
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

        public override void Update(IList<NdArray> labels, IList<NdArray> preds)
        {
            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].AsNumerics();
                var pred = preds[i].AsNumerics();

                if (pred.Shape[1] == 2)
                {
                    //TODO pred = pred[:, 1]
                }
                var reval = this._feval(label, pred);
                this.SumMetric[0] += reval.SumMetric;
                this.NumInst[0] += reval.NumInst;
            }
        }
    }
}