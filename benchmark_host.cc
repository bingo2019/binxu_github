#include <iostream>
#include <thread>
#include <fstream>
#include <future>
#include "de_node.h"
#include "de_registry.h"
#include "de_graph_api.h"
#include "de_model_api.h"
#include "de_aiengine_node.h"
#include "de_packed_func.h"
#include "fml_dsp_profile.h"

namespace de {

typedef struct {
	NDArray  array; 	///< data object
	int 	 id;	///< request id, response return same id
	uint64_t pts;		///< timestamp
  int end_flag = 1; ///< last packet flag

  inline de::NDArray Serialize() {
      int32_t *ptr = static_cast<int32_t*>(array.GetBearData());
      ptr[0] = id;
      ptr[1] = pts & 0xFFFFFFFF;
      ptr[2] = (pts >> 32) & 0xFFFFFFFF;
      ptr[3] = end_flag;
      std::vector<int64_t> shapes;
      array.Shrink(shapes, 4*sizeof(int32_t));
      return array;
  }

  inline void DeSerialize(de::NDArray data) {
      int32_t *ptr = static_cast<int32_t*>(data.GetBearData());
      id = ptr[0];
      pts = (((int64_t)ptr[2])<<32) | ((int64_t)ptr[1] & 0xFFFFFFFF);
      end_flag = ptr[3];
      array = data;
  }
    
}SenderTask;

namespace serializer {
STRUCT_SERIALIZE_4(SenderTask,
	NDArray, array,
	int, id,
	uint64_t, pts,
  int, end_flag);
}

/// @brief 发送者node，用于将数据发送给芯片
class Sender : public de::Thread
{
 public:
  Sender() {
    pin.SetTypeInfo(0, "de::SenderTask", de::TaskDeSerializeCustom<de::SenderTask>, de::TaskDeleter<de::SenderTask>);
    pouts_[0]->SetTypeInfo("de::SenderTask", de::TaskSerializeCustom<de::SenderTask>);
    SetName("sender");
  }
  ~Sender() {
    //删除入队列，停止算子
    pin.DestroyQueue();

    //等待队列线程退出
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  virtual void Proc(void* rx_task, int32_t task_type, POutType pout_type) {
    pouts_[0]->SendTask(rx_task);
  
  }
  void Stop(void){pin.DestroyQueue();}
};

/// @brief 接收者node，用于接收芯片侧结果返回
class Receiver : public de::Thread
{
 public:
  Receiver() {
    pin.SetTypeInfo(0, "de::SenderTask", de::TaskDeSerializeCustom<de::SenderTask>, de::TaskDeleter<de::SenderTask>);
    pouts_[0]->SetTypeInfo("de::NNTask", de::TaskSerialize<de::NNTask>);
    attr_.SetDefault("chip_test_mode", 0, 0, 1);
    SetName("receiver");
  }
  ~Receiver() {
    //删除入队列，停止算子
    pin.DestroyQueue();
    
    //等待队列线程退出
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void Stop(void){pin.DestroyQueue();}

  virtual void Proc(void* rx_task, int32_t task_type, POutType pout_type) {
    int chip_test_mode = 0;
    attr_.Get("chip_test_mode", chip_test_mode);
    if (task_type == 0) {
      auto task = reinterpret_cast<de::SenderTask*>(rx_task);
      // printf("<==== Enter Receiver Proc %x %x\n", task->id, task->pts);
      if (last_id_ != task->id) {
        nn_task_ = new de::NNTask();  
      }
      // if (chip_test_mode){
      //   nn_task_->id = (task->pts << 32)  + task->id;
      //   // printf("<==== nn_task_->id %x\n", nn_task_->id);
      // }
      // else
        nn_task_->id = task->id;
      nn_task_->tensors.emplace_back(task->array);
      if (task->end_flag) {
        // printf("<==== receive nntask id = %ld\n",nn_task_->id );
        pouts_[0]->SendTask(nn_task_);
      }
      last_id_ = task->id;
    }
    pin.DelTask(rx_task, task_type);
  }
  private:
  de::NNTask* nn_task_ = nullptr;
  int last_id_ = -1;
};

//注册全局node
DE_CLASS_REGISTER("de::Sender", Sender);
DE_CLASS_REGISTER("de::Receiver", Receiver);

struct PrecitPar
{
int dev_id;
  //推理次数
  int64_t predict_num;
  //最大batchsize 上限32
  int64_t batch_max;
  //当前batchsize
  int64_t batch_now;
  //是否硬件测试
  int chip_test_mode;
  //硬件测试单次结果
  //int64_t chip_test_result;
  float chip_test_fps;
  float chip_test_delay;
  //资源数
  int resource_num;
  //模型输入个数，上限4
  int tensor_in_one_batch = 1;
  //nnp 使用个数
  int nnp_bit_mask=4;
  de::Graph* graph;
  void* profile;
  //最后一张图片推理future
  std::future<int64_t> result;
  //保存每张图片推理结果（端到端）
  std::vector<de::NNTask> output_tasks;
  //时间戳（端到端）
  std::map<uint64_t, std::chrono::time_point<std::chrono::steady_clock>> start_ts;
  std::map<uint64_t, std::chrono::time_point<std::chrono::steady_clock>> end_ts;
};

DE_REGISTER_GLOBAL("de.benchmark.load_model").
	set_body([](de::DEArgs args, de::DERetValue *rv) {		
    //加载模型到芯片内存
    int dev_id = args[0];
    de::ModelManager mem(dev_id);
    std::string net_bin_path = args[1];
    std::string model_bin_path = args[2];
    std::string net_bin = "host:" + net_bin_path;
    std::string model_bin = "host:" + model_bin_path;
    mem.Load(net_bin, model_bin, false); 
	});

DE_REGISTER_GLOBAL("de.benchmark.unload_model").
	set_body([](de::DEArgs args, de::DERetValue *rv) {		
    int dev_id = args[0];
    //加载模型到芯片内存
    de::ModelManager mem(dev_id);
    std::string net_bin_path = args[1];
    std::string net_bin = "host:" + net_bin_path;
    mem.Unload(net_bin);
	});

DE_REGISTER_GLOBAL("de.benchmark.predict.init").
	set_body([](de::DEArgs args, de::DERetValue *rv) {
  std::string net_bin_path = args[0];
  std::string model_bin_path = args[1];
  std::string net_bin = "host:" + net_bin_path;
  std::string model_bin = "host:" + model_bin_path;

  PrecitPar* par = new PrecitPar;
  //测试图片个数
  par->predict_num = args[2];
  //最大batchsize，可以推入aie的nntask最大32组输入
  par->batch_max = args[3];
  //当前batchsize
  par->batch_now = args[4];
  //模型输入个数
  par->tensor_in_one_batch = args[5];
  //是否上报推理结果
  int report_flag = args[6];
  //是否芯片做解码
  int decode_onchip = args[7];
  //芯片硬件性能测试
  par->chip_test_mode = args[8];
  //配置资源数
  par->resource_num = args[9];
  //配置使用nnp个数
  par->nnp_bit_mask = args[10];
  int dev_id = args[11];
  int input_que_size = args[12];
  par->dev_id = dev_id;

  //创建graph
  de::Graph* graph = new de::Graph("model_pred");
  par->graph = graph;
  //创建bridge，连接取流和解码2个node
  // graph->CreateBridge(0, "192.168.145.130", 9200);
  graph->CreateBridge(dev_id);

  std::map<std::string, int> attr;
  // 聚包配置
  // attr["tx_pack_timeout"] = 200;
  // attr["rx_unpack"] = 1;
  
  //关闭crc校验
  attr["need_check_crc"] = 0;

  graph->AddBridgeH2DChan(0, "Host2Device", attr);
  graph->AddBridgeD2HChan(1, "Device2Host", attr);

  //创建node
  graph->CreateHostNode("de::Sender", "Sender");
  graph->CreateDevNode("de::AiEngineTest", "AiEngine");
  graph->CreateHostNode("de::Receiver", "Receiver");
  if (decode_onchip == 1) graph->CreateDevNode("de::JpegDecoder", "JpegDecNode");
  graph->CreateDevNode("de::DevReceiver", "DevReceiver");
  graph->CreateDevNode("de::DevSender", "DevSender");
  // printf("!!!!![%d] node create ok!\n",dev_id);
  //设置node属性
  graph->SetNodeAttr("AiEngine", "model_net_path", net_bin);
  graph->SetNodeAttr("AiEngine", "model_param_path", model_bin);
  graph->SetNodeAttr("AiEngine", "resize_type", decode_onchip);
  graph->SetNodeAttr("AiEngine", "batch_num", (int)par->batch_max);
  graph->SetNodeAttr("AiEngine", "resource_num", (int)par->resource_num);
  graph->SetNodeAttr("AiEngine", "nnp_bit_mask", (int)par->nnp_bit_mask);
  graph->SetNodeAttr("DevSender", "report_flag", report_flag);
  if (par->chip_test_mode==1)
    graph->SetNodeAttr("AiEngine", "pin.que.size", input_que_size);
  else
    graph->SetNodeAttr("Sender", "pin.que.size", input_que_size);
  // printf("!!!!![%d] node attr1 ok!\n",dev_id);

  //chip_test_mode为1时，DevSender发送fps结果回主控
  graph->SetNodeAttr("DevReceiver", "chip_test_mode", (int)par->chip_test_mode);
  graph->SetNodeAttr("DevReceiver", "predict_num", (int)par->predict_num);
  graph->SetNodeAttr("DevSender", "chip_test_mode", (int)par->chip_test_mode);
  graph->SetNodeAttr("DevSender", "predict_num", (int)par->predict_num);
  graph->SetNodeAttr("Receiver", "chip_test_mode", (int)par->chip_test_mode);
  //vector_btnum 配置nntask一次批量的输入，仅对芯片测试模式有效
  graph->SetNodeAttr("DevSender", "vector_btnum", (int)par->batch_now);
  graph->SetNodeAttr("DevReceiver", "vector_btnum", (int)par->batch_now);
  // printf("!!!!![%d] node attr2 ok!\n",dev_id);

  //设置graph内node连接
  graph->LinkNode("Sender", 0, "DevReceiver", 0);
  // printf("!!!!![%d] node link ok 0!\n",dev_id);

  if (decode_onchip == 1) {
    graph->LinkNode("DevReceiver", 0, "JpegDecNode");
    graph->LinkNode("JpegDecNode", 0, "AiEngine");
    graph->LinkNode("AiEngine", 1, "DevSender");
  }
  else{
    graph->LinkNode("DevReceiver", 2, "AiEngine");
// printf("!!!!![%d] node link ok!1\n",dev_id);

    graph->LinkNode("AiEngine", 1, "DevSender");
  }
// printf("!!!!![%d] node link ok! 2\n",dev_id);

  graph->LinkNode("DevSender", 0, "Receiver", 1);
  // printf("!!!!![%d] node link ok!\n",dev_id);

  //设置graph输入和输出
  graph->SetInputNode(0, "Sender");
  graph->SetOutputNode(0, "Receiver", 0);

  //启动业务
  graph->Start();
// printf("!!!!![%d] graph start ok!\n",dev_id);

  //启动dsp nnp调用率获取
  auto pf = de::Device::GetFunc(dev_id, "de.dspprofile.open");
  par->profile = pf(0xF);
  

  //启动结果读取线程，获取graph执行结果
  par->result = std::async(std::launch::async, [par](de::Graph *_graph) mutable {
    void *p;
    int64_t idx = 0;
    while (nullptr != (p = _graph->GetOutput(0))) {
      de::NNTask *task = static_cast<de::NNTask*>(p);
      LOG(ERROR) << "<====AiEngine result id " << task->id << " size " << task->tensors.size() << " tensor0size " <<task->tensors[0].GetTensorDataSize();
      if (par->chip_test_mode) {
        //par->chip_test_result = (int64_t)task->id;
        //LOG(ERROR) << "par->chip_test_result " << par->chip_test_result;
        float* p = (float*)task->tensors[0].GetTensorData();
        par->chip_test_fps = *p;
        par->chip_test_delay = *(p+1);
        delete task;
		_graph->StopOutput<de::NNTask>(0); 
        return idx;
      }
      else{
        idx = task->id;
        par->end_ts[task->id] = std::chrono::steady_clock::now();
        par->output_tasks.push_back(*task);
        delete task;
        if (idx >= par->predict_num){
		  _graph->StopOutput<de::NNTask>(0); 
          return idx;
        }
      }
    }
    return idx;
  }, graph);

  *rv = par; 
// printf("!!!!![%d] init ok!\n",dev_id);

});

DE_REGISTER_GLOBAL("de.benchmark.predict.delay").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
		int task_idx = args[1];
    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(par->end_ts[task_idx] - par->start_ts[task_idx]).count();
		*rv = duration;
});

DE_REGISTER_GLOBAL("de.benchmark.predict.outputnum").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
		int task_idx = args[1];
		*rv = (int64_t)par->output_tasks[task_idx].tensors.size();
});

DE_REGISTER_GLOBAL("de.benchmark.predict.output").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
		int task_idx = args[1];
    int nd_idx = args[2];
    // printf("%d outtasks  idx %d ndidx %d\n",par->output_tasks.size(),task_idx,nd_idx);
		*rv = par->output_tasks[task_idx].tensors[nd_idx];
});

DE_REGISTER_GLOBAL("de.benchmark.predict.output.del").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
	par->graph->StopOutput<de::NNTask>(0);
	int id = par->dev_id;
  
  //关闭dsp nnp调用率获取
  de::Device::GetFunc(id, "de.dspprofile.close")(par->profile, par->nnp_bit_mask);

	// printf("!!!!![%d] del 0!\n",par->dev_id);
	delete par->graph;
	// printf("!!!!![%d] del 1!\n",par->dev_id);
    delete par;
	// printf("!!!!![%d]del finish!\n",id);
});

DE_REGISTER_GLOBAL("de.benchmark.dspprofile.get").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
		int dsp_id = args[1];
    int type = args[2];

    std::string str = de::Device::GetFunc(par->dev_id, "de.dspprofile.get")(par->profile, type, dsp_id);
    std::vector<de::profile::DspProfileInfo> vinfo = de::fromByteArray<std::vector<de::profile::DspProfileInfo>>(str);
		*rv = vinfo[0].tot_sched_rate;
});

DE_REGISTER_GLOBAL("de.benchmark.predict.setinput").
	set_body([](de::DEArgs args, de::DERetValue *rv) {
  PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
  DEByteArray* byte_array = args[1].ptr<DEByteArray>();
  int64_t id = args[2];
  int64_t shape0 = args[3];
  int64_t shape1 = args[4];
  int64_t format = args[5];
  int64_t last_in_one = args[6];

  de::Graph *graph = par->graph;

  par->start_ts[id] = std::chrono::steady_clock::now();

  auto task = new de::SenderTask;
  NDArray ndarray = NDArray::Create(shape0, shape1, format, {0, 0, shape1, shape0}, const_cast<char*>(byte_array->data));

  task->array = ndarray;
  task->id = id;

  if (last_in_one == 0)
    task->end_flag = false;
  printf("sendertask id %d, task->end_flag %d\n",id, task->end_flag);
  graph->SetInput(0, task);
  
  if (par->chip_test_mode) {
    par->result.get();
    //*rv = par->chip_test_result;
  }
  else{
    if(id >= par->predict_num && last_in_one == 1) {
      par->result.get();
    }
  }
});

DE_REGISTER_GLOBAL("de.benchmark.get.chip.result").
		set_body([](DEArgs args, DERetValue *rv) {
    PrecitPar* par = static_cast<PrecitPar*>((void*)args[0]);
		int type = args[1];
    if (type == 0)
      *rv = par->chip_test_fps;
    else if (type == 1)
      *rv = par->chip_test_delay;
});


DE_REGISTER_GLOBAL("de.benchmark.load.library").
	set_body([](de::DEArgs args, de::DERetValue *rv) {
  int dev_id = args[0];
  std::string lib_path= args[1];
	void* handle = de::Device::LoadLib(dev_id,lib_path);
	*rv = handle;
});

DE_REGISTER_GLOBAL("de.benchmark.unload.library").
	set_body([](de::DEArgs args, de::DERetValue *rv) {
  int dev_id = args[0];
  void* handle = (void*)args[1];
  de::Device::UnLoadLibHandle(dev_id,handle);
});

}
