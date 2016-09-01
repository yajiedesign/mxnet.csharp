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
        private bool _is_dispose = false;
        public NameScop()
        {
            NameManager.Instance.Push();
        }

        public void Dispose()
        {
            if (_is_dispose == false)
            {
                NameManager.Instance.Pop();
                _is_dispose = true;
            }
        }
    }

    class NameUnit
    {
        private readonly Dictionary<string, int> _namedict = new Dictionary<string, int>();
        public string get_name(string op_name)
        {
            op_name = op_name.ToLower();
            if (!_namedict.ContainsKey(op_name))
            {
                _namedict.Add(op_name, 0);
            }
            _namedict[op_name]++;

            return $"{op_name}{_namedict[op_name]:D2}";
        }
    }

    class NameManager
    {
        private static readonly ThreadLocal<NameManager> Instancetls = new ThreadLocal<NameManager>(() => new NameManager());
        public static NameManager Instance => Instancetls.Value;

        private static readonly NameUnit Default = new NameUnit();
        private readonly Stack<NameUnit> _stack = new Stack<NameUnit>();

        private NameManager()
        {

        }

        public string get_name(string op_name)
        {
            if (_stack.Count == 0)
            {
                lock (Default)
                {
                    return Default.get_name(op_name);
                }
            }
            return _stack.Peek().get_name(op_name);
        }

        public void Push()
        {
            _stack.Push(new NameUnit());
        }

        public void Pop()
        {
            _stack.Pop();
        }
    }
}
