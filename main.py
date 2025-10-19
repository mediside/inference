from concurrent import futures
import grpc
from proto import inference_pb2
from proto import inference_pb2_grpc
import inference

GRPC_HOST = '0.0.0.0'
GRPC_PORT = 30042

class InferenceService(inference_pb2_grpc.InferenceServicer):
    def DoInference(self, request, context):
        try:
            for percent, step in inference.doInference(request.file_path, request.study_id, request.series_id):
                if percent < 100:
                    progress = inference_pb2.Progress(percent=percent, step=step)
                    response = inference_pb2.InferenceResponse(progress=progress)
                else:
                    result = inference_pb2.Result(probability_of_pathology=step)
                    response = inference_pb2.InferenceResponse(result=result)
                yield response
        except Exception as e:
            print("Error in DoInference:", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
    server.start()
    print(f'gRPC server started on port {GRPC_PORT}')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()