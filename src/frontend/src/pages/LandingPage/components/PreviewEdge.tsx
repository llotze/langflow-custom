import { BaseEdge, type EdgeProps, getBezierPath, Position } from "@xyflow/react";

// Simple edge component for preview that doesn't rely on flow store
export function PreviewEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  ...props
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    targetX,
    targetY,
  });

  const { animated, selectable, deletable, selected, ...domSafeProps } = props;

  return (
    <BaseEdge
      path={edgePath}
      {...domSafeProps}
      data-animated={animated ? "true" : "false"}
      data-selectable={selectable ? "true" : "false"}
      data-deletable={deletable ? "true" : "false"}
      data-selected={selected ? "true" : "false"}
    />
  );
}
