if step <= 1:
                    t0 = time.perf_counter()
                    # move dataset to device
                    current_file_SVC_org = current_file_SVC_org.to(device, non_blocking=True)
                    current_file_SVC_thickGt = current_file_SVC_thickGt.to(device, non_blocking=True)
                    current_file_SVC_thinGt = current_file_SVC_thinGt.to(device, non_blocking=True)
                    t1 = time.perf_counter()
                    if scaler is not None:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            test_out, test_thinout = net(current_file_SVC_org)
                        print(test_out.dtype)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    print(f"step {step}: "
                          f"CPU→GPU {(t1 - t0)*1000:.1f} ms | "
                          f"forward {(t2 - t1)*1000:.1f} ms", flush=True)
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        _ = net(current_file_SVC_org)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    print(f"step {step}: "
                          f"CPU→GPU {(t1 - t0)*1000:.1f} ms | "
                          f"forward {(t2 - t1)*1000:.1f} ms", flush=True)