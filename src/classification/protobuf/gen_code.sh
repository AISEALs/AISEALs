protoc -I . --python_out=. feature.proto
protoc -I . --python_out=. example.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. dlserver.proto
