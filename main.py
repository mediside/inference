from concurrent import futures
import grpc
from proto import inference_pb2
from proto import inference_pb2_grpc
import inference

GRPC_HOST = 'localhost'
GRPC_PORT = 30042

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def DoInference(self, request, context):
        for percent, step in inference.doInference(request.file_path):
            if percent < 100:
                progress = inference_pb2.Progress(percent=percent, step=step)
                response = inference_pb2.InferenceResponse(progress=progress)
            else:
                result = inference_pb2.Result(probability_of_pathology=step)
                response = inference_pb2.InferenceResponse(result=result)
            yield response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
    server.start()
    print(f'gRPC server started on port {GRPC_PORT}')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()