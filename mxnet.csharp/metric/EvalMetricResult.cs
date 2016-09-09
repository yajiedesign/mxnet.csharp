namespace mxnet.csharp.metric
{
    public class EvalMetricResult
    {


        public EvalMetricResult(string name, float value)
        {
            this.Name = name;
            this.Value = value;
        }

        public string Name { get;  }
        public float Value { get;}
    }
}