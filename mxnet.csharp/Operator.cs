using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SymbolHandle = System.IntPtr;
using AtomicSymbolCreator = System.IntPtr;
namespace mxnet.csharp
{
    public partial class Operator 
    {
        private readonly string _operator_name;
        static readonly OpMap OpMap = new OpMap();
        readonly Dictionary<string, string> _params_ = new Dictionary<string, string>();
        readonly List<SymbolHandle> _input_values = new List<SymbolHandle>();
        private readonly List<string> _input_keys = new List<string>();
        private readonly AtomicSymbolCreator _handle_;

        /// <summary>
        /// Operator constructor
        /// </summary>
        /// <param name="operator_name">type of the operator</param>
        public Operator(string operator_name)
        {
            _operator_name = operator_name;
   
            _handle_ = OpMap.GetSymbolCreator(operator_name);
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
            _params_[name] = value.ToString();
            return this;
        }

        /// <summary>
        /// add an input symbol
        /// </summary>
        /// <param name="name">name name of the input symbol</param>
        /// <param name="symbol">the input symbol</param>
        /// <returns></returns>
        public Operator SetInput(string name, Symbol symbol)
        {
            if (symbol == null)
            {
                return this;
            }
            _input_keys.Add(name);
            _input_values.Add(symbol.GetHandle());
            return this;
        }
        public Operator AddInput(Symbol s1)
        {
            PushInput(s1);
            return this;
        }
        public Operator AddInput(ICollection<Symbol> sc)
        {
            foreach (var s in sc)
            {
                PushInput(s);
            }
            return this;
        }

        public Operator AddInput(Symbol s1, Symbol s2)
        {
            PushInput(s1);
            PushInput(s2);
            return this;
        }
        /*!
        * \brief add an input symbol
        * \param symbol the input symbol
        */
        /// <summary>
        /// add an input symbol
        /// </summary>
        /// <param name="symbol">the input symbol</param>
        public void PushInput(Symbol symbol)
        {
            _input_values.Add(symbol.GetHandle());
        }




        /// <summary>
        /// create a Symbol from the current operator
        /// </summary>
        /// <param name="name">the name of the operator</param>
        /// <returns>the operator Symbol</returns>
        public Symbol CreateSymbol(string name = "")
        {
            string pname = name == "" ? NameManager.instance.get_name(_operator_name) : name;

            SymbolHandle symbol_handle;
            List<string> input_keys = new List<string>();
            List<string> param_keys = new List<string>();
            List<string> param_values = new List<string>();

            foreach (var data in _params_)
            {
                param_keys.Add(data.Key);
                param_values.Add(data.Value);
            }
            foreach (var data in this._input_keys)
            {
                input_keys.Add(data);
            }



            NativeMethods.MXSymbolCreateAtomicSymbol(_handle_, (uint)param_keys.Count, param_keys.ToArray(),
                                       param_values.ToArray(), out symbol_handle);

            if (input_keys.Count > 0)
            {
                if (NativeMethods.MXSymbolCompose(symbol_handle, pname, (uint) _input_values.Count, input_keys.ToArray(),
                    _input_values.ToArray()) != 0)
                {
                    string error = (NativeMethods.MXGetLastError());
                }
            }
            else
            {
                if (NativeMethods.MXSymbolCompose(symbol_handle, pname, (uint) _input_values.Count, IntPtr.Zero,
                    _input_values.ToArray())==0)
                {
                    string error = (NativeMethods.MXGetLastError());
                }
            }

            return new Symbol(symbol_handle);
        }


    }
}
