using System;

namespace mxnet.csharp.metric
{
    class CustomMetric : EvalMetric
    {
        public CustomMetric
            (Func<object,object,float> eval , string name =null, int num = 1) : base(GetName(eval,name), num)
        {
        }

        private static string GetName(Func<object, object, float> eval, string name)
        {
            return !string.IsNullOrWhiteSpace(name) ? name : $" Custom({eval.GetType().FullName})";
        }

        public override void update(object label, object pred)
        {
            throw new NotImplementedException();
        }
    }
}