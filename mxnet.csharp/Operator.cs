using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SymbolHandle = System.IntPtr;
using NdArrayHandle = System.IntPtr;
using AtomicSymbolCreator = System.IntPtr;
namespace mxnet.csharp
{
    public partial class Operator 
    {
        private readonly string _operatorName;
        static readonly OpMap OpMap = new OpMap();
        readonly Dictionary<string, string> _params = new Dictionary<string, string>();
        readonly List<SymbolHandle> _inputValues = new List<SymbolHandle>();
        private readonly IList<string> _inputKeys = new List<string>();
        private readonly AtomicSymbolCreator _handle;

        /// <summary>
        /// Operator constructor
        /// </summary>
        /// <param name="operatorName">type of the operator</param>
        public Operator(string operatorName)
        {
            _operatorName = operatorName;
   
            _handle = OpMap.GetSymbolCreator(operatorName);
        }

        /// <summary>
        /// set config parameters
        /// </summary>
        /// <typeparam name="TT"></typeparam>
        /// <param name="name">name of the config parameter</param>
        /// <param name="value">value of the config parameter</param>
        /// <returns></returns>
        public Operator SetParam<TT>(string name, TT value)
        {
            if (value == null)
            {
                return this;
            }
            _params[name] = value.ToString();
            return this;
        }

        /// <summary>
        /// add an input symbol
        /// </summary>
        /// <param name="name">name name of the input symbol</param>
        /// <param name="symbol">the input symbol</param>
        /// <returns></returns>
        public Operator SetInput(string name, INdArrayOrSymbol symbol)
        {
            if (symbol == null)
            {
                return this;
            }
            _inputKeys.Add(name);
            _inputValues.Add(symbol.get_handle());
            return this;
        }
        public Operator AddInput(INdArrayOrSymbol s1)
        {
            PushInput(s1);
            return this;
        }
        public Operator AddInput(ICollection<INdArrayOrSymbol> sc)
        {
            foreach (var s in sc)
            {
                PushInput(s);
            }
            return this;
        }

        public Operator AddInput(INdArrayOrSymbol s1, INdArrayOrSymbol s2)
        {
            PushInput(s1);
            PushInput(s2);
            return this;
        }

        /// <summary>
        /// add an input symbol
        /// </summary>
        /// <param name="symbol">the input symbol</param>
        public void PushInput(INdArrayOrSymbol symbol)
        {
            _inputValues.Add(symbol.get_handle());
        }

 



        /// <summary>
        /// create a Symbol from the current operator
        /// </summary>
        /// <param name="name">the name of the operator</param>
        /// <returns>the operator Symbol</returns>
        public Symbol CreateSymbol(string name = "")
        {
            string pname = name == "" ? NameManager.Instance.GetName(_operatorName) : name;

            SymbolHandle symbolHandle;
            List<string> inputKeys = new List<string>();
            List<string> paramKeys = new List<string>();
            List<string> paramValues = new List<string>();

            foreach (var data in _params)
            {
                paramKeys.Add(data.Key);
                paramValues.Add(data.Value);
            }
            foreach (var data in this._inputKeys)
            {
                inputKeys.Add(data);
            }



            Util.CallCheck(NativeMethods.MXSymbolCreateAtomicSymbol(_handle, (uint)paramKeys.Count, paramKeys.ToArray(),
                                        paramValues.ToArray(), out symbolHandle));

            if (inputKeys.Count > 0)
            {
                if (NativeMethods.MXSymbolCompose(symbolHandle, pname, (uint) _inputValues.Count, inputKeys.ToArray(),
                    _inputValues.ToArray()) != 0)
                {
                    string error = (NativeMethods.MxGetLastError());
                }
            }
            else
            {
                if (NativeMethods.MXSymbolCompose(symbolHandle, pname, (uint) _inputValues.Count, IntPtr.Zero,
                    _inputValues.ToArray())==0)
                {
                    string error = (NativeMethods.MxGetLastError());
                }
            }

            return new Symbol(symbolHandle);
        }

        public void Invoke(List<NdArray> outputs)
        {

   
            List<string> inputKeys = new List<string>();
            List<string> paramKeys = new List<string>();
            List<string> paramValues = new List<string>();

            foreach (var data in _params)
            {
                paramKeys.Add(data.Key);
                paramValues.Add(data.Value);
            }
            foreach (var data in this._inputKeys)
            {
                inputKeys.Add(data);
            }
            int num_inputs = _inputValues.Count;
            int num_outputs = outputs.Count;

            NdArrayHandle[] output_handles = outputs.Select(s => s.get_handle()).ToArray();
            IntPtr outputs_receiver = IntPtr.Zero;
            GCHandle? gcHandle = null;
            if (outputs.Count > 0)
            {
                gcHandle = GCHandle.Alloc(output_handles, GCHandleType.Pinned);
                outputs_receiver = gcHandle.Value.AddrOfPinnedObject();

            }

            NativeMethods.MXImperativeInvoke(_handle, num_inputs, _inputValues.ToArray(),ref num_outputs, ref outputs_receiver,
                paramKeys.Count, paramKeys.ToArray(), paramValues.ToArray());

            if (outputs.Count > 0)
            {
                gcHandle?.Free();
                return;
            }
            output_handles = new IntPtr[num_outputs];

            Marshal.Copy(outputs_receiver, output_handles,0, num_outputs);

            foreach (IntPtr outputHandle in output_handles)
            {
                outputs.Add(new NdArray(outputHandle));
            }

        }

        public NdArray Invoke(NdArray @out)
        {
            List<NdArray> outputs = new List<NdArray> { @out };
            Invoke(outputs);
            return outputs.First();
        }

        public NdArray Invoke()
        {
            List<NdArray> outputs = new List<NdArray> {new NdArray()};
            Invoke(outputs);
            return outputs.First();
        }

    }
}
