namespace mxnet.csharp.metric
{
    public class EvalMetricResult
    {


        public EvalMetricResult(string name, float value)
        {
            Name = name;
            Value = value;
        }

        public string Name { get;  }
        public float Value { get;}
    }
}