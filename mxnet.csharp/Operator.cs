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
        static readonly OpMap op_map_ = new OpMap();
        Dictionary<string, string> params_desc_ = new Dictionary<string, string>();
        bool variable_params_ = false;
        Dictionary<string, string> params_ = new Dictionary<string, string>();
        List<SymbolHandle> input_values = new List<SymbolHandle>();
        List<string> input_keys = new List<string>();
        private AtomicSymbolCreator handle_;

        /// <summary>
        /// Operator constructor
        /// </summary>
        /// <param name="operator_name">type of the operator</param>
        public Operator(string operator_name)
        {
            handle_ = op_map_.GetSymbolCreator(operator_name);
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

            params_[name] = value.ToString();
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
            input_keys.Add(name);
            input_values.Add(symbol.GetHandle());
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
        void PushInput(Symbol symbol)
        {
            input_values.Add(symbol.GetHandle());
        }




        /// <summary>
        /// create a Symbol from the current operator
        /// </summary>
        /// <param name="name">the name of the operator</param>
        /// <returns>the operator Symbol</returns>
        public Symbol CreateSymbol(string name = "")
        {
            string pname = name == "" ? null : name;

            SymbolHandle symbolHandle;
            List<string> inputKeys = new List<string>();
            List<string> paramKeys = new List<string>();
            List<string> paramValues = new List<string>();

            foreach (var data in params_)
            {
                paramKeys.Add(data.Key);
                paramValues.Add(data.Value);
            }
            foreach (var data in this.input_keys)
            {
                inputKeys.Add(data);
            }



            NativeMethods.MXSymbolCreateAtomicSymbol(handle_, (uint)paramKeys.Count, paramKeys.ToArray(),
                                       paramValues.ToArray(), out symbolHandle);

            if (inputKeys.Count > 0)
            {
                NativeMethods.MXSymbolCompose(symbolHandle, pname, (uint)input_values.Count, inputKeys.ToArray(),
                    input_values.ToArray());
            }
            else
            {
                NativeMethods.MXSymbolCompose(symbolHandle, pname, (uint)input_values.Count, IntPtr.Zero,
                    input_values.ToArray());
            }

            return new Symbol(symbolHandle);
        }


    }
}
