using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace mxnet.csharp
{

    public class NameScop : IDisposable
    {
        private bool is_dispose = false;
        public NameScop()
        {
            NameManager.instance.push();
        }

        public void Dispose()
        {
            if (is_dispose == false)
            {
                NameManager.instance.pop();
                is_dispose = true;
            }
        }
    }

    class NameUnit
    {
        private readonly Dictionary<string, int> namedict = new Dictionary<string, int>();
        public string get_name(string op_name)
        {
            op_name = op_name.ToLower();
            if (!namedict.ContainsKey(op_name))
            {
                namedict.Add(op_name, 0);
            }
            namedict[op_name]++;

            return $"{op_name}{namedict[op_name]:D2}";
        }
    }

    class NameManager
    {
        private static readonly ThreadLocal<NameManager> Instancetls = new ThreadLocal<NameManager>(() => new NameManager());
        public static NameManager instance => Instancetls.Value;

        private static readonly NameUnit Default = new NameUnit();
        private readonly Stack<NameUnit> stack = new Stack<NameUnit>();

        private NameManager()
        {

        }

        public string get_name(string op_name)
        {
            if (stack.Count == 0)
            {
                lock (Default)
                {
                    return Default.get_name(op_name);
                }
            }
            return stack.Peek().get_name(op_name);
        }

        public void push()
        {
            stack.Push(new NameUnit());
        }

        public void pop()
        {
            stack.Pop();
        }
    }
}
