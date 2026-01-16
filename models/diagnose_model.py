"""
Diagnostic script to identify gradient explosion sources in CrabTransformer
"""

import torch
import torch.nn as nn


def diagnose_model(model, sample_input):
    """
    Check model for common gradient explosion issues.
    
    Args:
        model: Your CrabTransformer instance
        sample_input: Sample tensor [batch, channels, height, width]
    """
    print("="*70)
    print("MODEL DIAGNOSTICS")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. Check model components exist
    # -------------------------------------------------------------------------
    print("\n[1] Checking model components...")
    
    required = ['patch_embed', 'position_encode', 'transformer_blocks', 'decoder']
    for name in required:
        if hasattr(model, name):
            print(f"  ✓ {name} exists")
        else:
            print(f"  ❌ {name} MISSING!")
    
    # -------------------------------------------------------------------------
    # 2. Check for LayerNorm in transformer blocks
    # -------------------------------------------------------------------------
    print("\n[2] Checking for LayerNorm in transformer blocks...")
    
    has_layernorm = False
    for i, block in enumerate(model.transformer_blocks):
        layernorms = [m for m in block.modules() if isinstance(m, nn.LayerNorm)]
        if layernorms:
            print(f"  ✓ Block {i}: {len(layernorms)} LayerNorm layers")
            has_layernorm = True
        else:
            print(f"  ❌ Block {i}: NO LayerNorm layers (CRITICAL!)")
    
    if not has_layernorm:
        print("\n  ⚠️  WARNING: No LayerNorm found in ANY transformer block!")
        print("     This WILL cause gradient explosion.")
        print("     Each transformer block should have 2 LayerNorm layers.")
    
    # -------------------------------------------------------------------------
    # 3. Check weight initialization
    # -------------------------------------------------------------------------
    print("\n[3] Checking weight initialization...")
    
    # Check a few weights to see if they're initialized properly
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if linear_layers:
        first_layer = linear_layers[0]
        weight_std = first_layer.weight.std().item()
        weight_mean = first_layer.weight.mean().item()
        
        print(f"  First linear layer weight stats:")
        print(f"    Mean: {weight_mean:.6f}")
        print(f"    Std:  {weight_std:.6f}")
        
        # Check if weights look initialized
        if abs(weight_mean) < 0.001 and 0.01 < weight_std < 0.5:
            print(f"  ✓ Weights appear properly initialized")
        else:
            print(f"  ⚠️  Unusual weight distribution - check initialization")
    
    # -------------------------------------------------------------------------
    # 4. Test forward pass and check for issues
    # -------------------------------------------------------------------------
    print("\n[4] Testing forward pass...")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            
            print(f"  ✓ Forward pass successful")
            print(f"  Input shape:  {sample_input.shape}")
            print(f"  Output shape: {output.shape}")
            
            # Check output statistics
            out_mean = output.mean().item()
            out_std = output.std().item()
            out_min = output.min().item()
            out_max = output.max().item()
            
            print(f"\n  Output statistics:")
            print(f"    Mean: {out_mean:.2f}")
            print(f"    Std:  {out_std:.2f}")
            print(f"    Min:  {out_min:.2f}")
            print(f"    Max:  {out_max:.2f}")
            
            # Check for issues
            if torch.isnan(output).any():
                print(f"  ❌ Output contains NaN!")
            if torch.isinf(output).any():
                print(f"  ❌ Output contains Inf!")
            if abs(out_mean) > 1000:
                print(f"  ⚠️  Output mean is very large: {out_mean:.2f}")
            if out_std > 1000:
                print(f"  ⚠️  Output std is very large: {out_std:.2f}")
            
        except Exception as e:
            print(f"  ❌ Forward pass FAILED: {str(e)}")
    
    # -------------------------------------------------------------------------
    # 5. Test gradient flow
    # -------------------------------------------------------------------------
    print("\n[5] Testing gradient flow...")
    
    model.train()
    sample_input.requires_grad = True
    
    try:
        output = model(sample_input)
        loss = output.mean()  # Dummy loss
        loss.backward()
        
        # Check gradient norms
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm > 100:
                    print(f"  ⚠️  Large gradient in {name}: {grad_norm:.2f}")
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        
        print(f"\n  Gradient statistics:")
        print(f"    Average gradient norm: {avg_grad_norm:.2f}")
        print(f"    Maximum gradient norm: {max_grad_norm:.2f}")
        
        if max_grad_norm > 100:
            print(f"  ❌ GRADIENT EXPLOSION DETECTED!")
            print(f"     Max gradient norm: {max_grad_norm:.2f}")
        elif max_grad_norm > 10:
            print(f"  ⚠️  Gradients are large: {max_grad_norm:.2f}")
        else:
            print(f"  ✓ Gradients are reasonable")
            
    except Exception as e:
        print(f"  ❌ Gradient test FAILED: {str(e)}")
    
    # -------------------------------------------------------------------------
    # 6. Component-specific checks
    # -------------------------------------------------------------------------
    print("\n[6] Checking decoder architecture...")
    
    decoder = model.decoder
    decoder_layers = list(decoder.modules())
    
    # Check for upsampling layers
    has_upsample = any(isinstance(m, (nn.ConvTranspose2d, nn.Upsample)) 
                       for m in decoder_layers)
    if has_upsample:
        print(f"  ✓ Decoder has upsampling layers")
    else:
        print(f"  ⚠️  Decoder may not have proper upsampling")
    
    # Check final activation
    has_activation = any(isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)) 
                        for m in decoder_layers)
    if has_activation:
        print(f"  ✓ Decoder has activation layers")
    else:
        print(f"  ⚠️  Decoder may not have activations")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues = []
    
    if not has_layernorm:
        issues.append("❌ CRITICAL: No LayerNorm in transformer blocks")
    
    if max_grad_norm > 100:
        issues.append("❌ CRITICAL: Gradient explosion detected")
    elif max_grad_norm > 10:
        issues.append("⚠️  WARNING: Gradients are large")
    
    if torch.isnan(output).any() or torch.isinf(output).any():
        issues.append("❌ CRITICAL: Output contains NaN or Inf")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\nRECOMMENDATIONS:")
        if not has_layernorm:
            print("  1. Add LayerNorm to transformer blocks")
        if max_grad_norm > 10:
            print("  2. Add proper weight initialization (_init_weights)")
            print("  3. Use gradient clipping (max_norm=1.0)")
            print("  4. Consider lower learning rate (1e-5)")
    else:
        print("\n✓ No major issues detected!")
    
    print("="*70)


def check_transformer_block_structure(transformer_block):
    """
    Check if a transformer block has proper structure.
    
    Expected structure:
    1. LayerNorm → Attention → Residual
    2. LayerNorm → FeedForward → Residual
    """
    print("\nTransformerBlock structure:")
    for name, module in transformer_block.named_children():
        print(f"  {name}: {module.__class__.__name__}")


# Example usage:
if __name__ == "__main__":
    # You would run this in your notebook/script like:
    
    # from your_module import CrabTransformer
    # 
    # model = CrabTransformer(...)
    # sample_input = torch.randn(2, 2, 50, 50)  # [batch, channels, height, width]
    # 
    # diagnose_model(model, sample_input)
    
    print("Copy this script to your notebook and run:")
    print("  diagnose_model(model, sample_input)")