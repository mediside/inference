proto_dir         = proto
proto_build_dir   = gen

compile-proto:
	python -m grpc_tools.protoc -I. \
	--proto_path=. \
	--python_out=. \
	--grpc_python_out=. \
	$(proto_dir)/*.proto
