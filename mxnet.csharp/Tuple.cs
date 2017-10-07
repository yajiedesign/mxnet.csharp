using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class Tuple<TType>
    {
        private readonly List<TType> _tuple = new List<TType>();
        public Tuple(params TType[] param)
        {
            _tuple.AddRange(param);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            _tuple.ForEach(f =>
            {
                sb.Append(f);
                sb.Append(",");
            });
            sb.Append(")");
            return sb.ToString();
        }
    }
}
