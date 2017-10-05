using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class Arg
    {
        private Dictionary<string, string> _typeDict = new Dictionary<string, string>
        {
            {"boolean", "bool"},
            {"Shape(tuple)", "Shape"},
            {"Symbol", "Symbol"},
            {"Symbol[]", "Symbol[]"},
            {"float", "float"},
            {"double", "double"},
            {"int", "int"},
            {"long", "long"},
            {"string", "string"},
            {"NDArray", "Symbol"},
            {"NDArray-or-Symbol", "NdArrayOrSymbol"},
            {"NDArray-or-Symbol[]", "NdArrayOrSymbol[]"}

        };
        private Dictionary<string, string> _typeDictWithNull = new Dictionary<string, string>
        {
            {"boolean", "bool"},
            {"float", "float"},
            {"double", "double"},
            {"int", "int"},
            {"long", "long"},
        };
        public string OrginName { get; }
        public string Name { get; }
        public string Description { get; }
        public bool IsEnum { get; } = false;
        public EnumType Enum { get; }
        public string TypeName { get; }
        public bool HasDefault { get; } = false;
        public string DefaultString { get; }
        public string DefaultStringWithObject { get; } = "";

        public Arg()
        {
            
        }

        public Arg(string opName = "", string argName = "", string typeString = "", string descString = "")
        {
            if (argName == "src")
            {
                argName = "data";
            }
            this.OrginName = argName;
            this.Name = GetName(argName);
            this.Description = descString;
            if (typeString.StartsWith("{"))
            {
                IsEnum = true;
                Enum = new EnumType(opName +"_" + argName, typeString);
                TypeName = Enum.Name;
            }
            else
            {
                string typename;

                if (_typeDict.TryGetValue(typeString.Split(' ').First().Replace(",", ""), out typename))
                {
                    TypeName = typename;
                }
                else
                {
                    if (opName == "Reshape")
                    {
                        TypeName = "Shape";
                    }
                }

             
            }
            if (typeString.IndexOf("default=", StringComparison.Ordinal) != -1)
            {
                HasDefault = true;
                DefaultString = typeString.Split(new string[] { "default=" }, StringSplitOptions.None)[1].Trim()
                    .Trim('\'');

                if (IsEnum)
                {
                    if (DefaultString == "None")
                    {
                        TypeName += "?";
                        DefaultString = "null";
                    }
                    else
                    {
                        DefaultString = Enum.GetDefaultValueString(DefaultString);

                    }

                }
                else if (DefaultString == "False")
                {
                    DefaultString = "false";
                }
                else if (DefaultString == "True")
                {
                    DefaultString = "true";
                }
                else if (DefaultString == "None")
                {
                    DefaultString = "null";
                    if (_typeDictWithNull.ContainsKey(TypeName))
                    {
                        TypeName += "?";
                    }
                }
                else if (DefaultString.StartsWith("("))
                {
                    if (DefaultString != "()")
                    {
                        DefaultStringWithObject = $"if({Name}==null){{ {Name}= new Shape{DefaultString};}}\n";
                    }
     
                    DefaultString = "null";       
                }
                if (TypeName == "float")
                {
                    DefaultString = DefaultString + "f";
                }
                if (TypeName == "string")
                {
                    DefaultString = "null";
                }
            }
            if (argName == "weight" || argName == "bias")
            {
                HasDefault = true;
                DefaultString = "null";
            }

        }

        private string GetName(string argName)
        {
            return argName;
        }
    }
    
}
