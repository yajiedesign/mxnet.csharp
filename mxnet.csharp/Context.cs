using System;
using System.Diagnostics;

namespace mxnet.csharp
{
    public enum DeviceType
    {
        KCpu = 1,
        KGpu = 2,
        KCpuPinned = 3
    };


    public class Context
    {
        /// <summary>
        /// Context constructor
        /// </summary>
        /// <param name="type">type of the device</param>
        /// <param name="id">id of the device</param>
        public Context(DeviceType type, int id)
        {
            DeviceType = type;
            DeviceId = id;
        }

        public static Context DefaultCtx { get; set; } = new Context(DeviceType.KCpu, 0);

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the type of the device</returns>
        [DebuggerHidden]
        public DeviceType DeviceType { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the id of the device</returns>
        [DebuggerHidden]
        public int DeviceId { get; }


        /// <summary>
        /// Return a GPU context
        /// </summary>
        /// <param name="deviceId">id of the device</param>
        /// <returns>the corresponding GPU context</returns>
        public static Context Gpu(int deviceId = 0)
        {
            return new Context(DeviceType.KGpu, deviceId);
        }


        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="deviceId">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int deviceId = 0)
        {
            return new Context(DeviceType.KCpu, deviceId);
        }

        public override string ToString()
        {
            switch (DeviceType)
            {
                case DeviceType.KCpu:
                    return $"cpu({DeviceId})";
                case DeviceType.KGpu:
                    return $"gpu({DeviceId})";
                case DeviceType.KCpuPinned:
                    return $"cpu_pinned({DeviceId})";
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    };
}