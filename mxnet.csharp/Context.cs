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
            device_type = type;
            device_id = id;
        }

        public static Context default_ctx { get; set; } = new Context(DeviceType.KCpu, 0);

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the type of the device</returns>
        [DebuggerHidden]
        public DeviceType device_type { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the id of the device</returns>
        [DebuggerHidden]
        public int device_id { get; }


        /// <summary>
        /// Return a GPU context
        /// </summary>
        /// <param name="device_id">id of the device</param>
        /// <returns>the corresponding GPU context</returns>
        public static Context Gpu(int device_id = 0)
        {
            return new Context(DeviceType.KGpu, device_id);
        }


        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="device_id">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int device_id = 0)
        {
            return new Context(DeviceType.KCpu, device_id);
        }



    };
}