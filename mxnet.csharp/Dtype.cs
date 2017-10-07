using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class Dtype
    {
        private static readonly Dictionary<string, Dtype> StringToDtypeMap = new Dictionary<string, Dtype>();
        private static readonly Dictionary<int, Dtype> IndexToDtypeMap = new Dictionary<int, Dtype>();
        public static readonly Dtype Float32 = new Dtype("float32", "Float32", 0);
        public static readonly Dtype Float64 = new Dtype("float64", "Float64", 1);
        public static readonly Dtype Float16 = new Dtype("float16", "Float16", 2);
        public static readonly Dtype Uint8 = new Dtype("uint8", "Uint8", 3);
        public static readonly Dtype Int32 = new Dtype("int32", "Int32", 4);
        public static readonly Dtype Int8 = new Dtype("int8", "Int8", 5);
        public static readonly Dtype Int64 = new Dtype("int64", "Int64", 6);

        public string Name { get; }
        public string CsName { get; }
        public int Index { get; }

        public Dtype(string name, string csName, int index)
        {
            Name = name;
            CsName = csName;
            Index = index;
            StringToDtypeMap.Add(Name, this);
            IndexToDtypeMap.Add(index, this);
        }
        public static implicit operator string(Dtype value)
        {
            return value.Name;
        }
        public static implicit operator Dtype(string value)
        {
            return StringToDtypeMap[value];
        }
        public static explicit operator Dtype(int index)
        {
            return IndexToDtypeMap[index];
        }

        public override string ToString()
        {
            return Name;
        }
    }
}
