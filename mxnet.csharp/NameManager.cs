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
        private bool _isDispose = false;
        public NameScop()
        {
            NameManager.Instance.Push();
        }

        public void Dispose()
        {
            if (_isDispose == false)
            {
                NameManager.Instance.Pop();
                _isDispose = true;
            }
        }
    }

    class NameUnit
    {
        private readonly Dictionary<string, int> _namedict = new Dictionary<string, int>();
        public string GetName(string opName)
        {
            opName = opName.ToLower();
            if (!_namedict.ContainsKey(opName))
            {
                _namedict.Add(opName, 0);
            }
            _namedict[opName]++;

            return $"{opName}{_namedict[opName]:D2}";
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

        public string GetName(string opName)
        {
            if (_stack.Count == 0)
            {
                lock (Default)
                {
                    return Default.GetName(opName);
                }
            }
            return _stack.Peek().GetName(opName);
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
