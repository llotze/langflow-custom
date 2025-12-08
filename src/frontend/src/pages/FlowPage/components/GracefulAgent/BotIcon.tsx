import { motion } from 'framer-motion';
import gracefulRobotHead from '@/assets/graceful/graceful-robot-head.png';

interface BotIconProps {
onBotClick: () => void;
isChatOpen?: boolean;
onClose?: () => void;
}

export function BotIcon({ onBotClick, isChatOpen = false, onClose }: BotIconProps) {
const handleClick = () => {
    if (isChatOpen && onClose) {
        onClose();
    } else {
        onBotClick();
    }
};

return (
<div className="absolute top-[54px] right-[235px] z-20 flex items-center">
    {/* Bot Icon - spins like a coin every 20 seconds */}
    <motion.button
    onClick={handleClick}
    className="w-12 h-12 bg-transparent p-0 border-0 cursor-pointer"
    animate={{
        rotateY: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0],
    }}
    transition={{
        duration: 20,
        repeat: Infinity,
        ease: "linear",
        times: [0, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
    }}
    >
    <img 
        src={gracefulRobotHead} 
        alt="Graceful Robot" 
        className="w-full h-full object-contain"
    />
    </motion.button>
</div>
);
}
