// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function click(name: string | RegExp) {
    fireEvent.click(screen.getByRole('button', { name }));
}

describe('App flow', () => {
    beforeEach(() => {
        window.localStorage.clear();
        vi.useFakeTimers();
        vi.spyOn(window, 'prompt').mockImplementation((_m, def) => def ?? '');
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    afterEach(() => {
        cleanup();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('generates, saves, and plays a workout', () => {
        render(<App />);
        click('哑铃');
        click('生成训练');

        expect(screen.getByText('热身')).toBeTruthy();
        expect(screen.getByText(/^主训练/)).toBeTruthy();
        expect(screen.getByText('放松')).toBeTruthy();

        click('保存为模板');
        expect(JSON.parse(window.localStorage.getItem('gym.saved.v1')!)).toHaveLength(1);

        click('▶ 开始跟练');
        expect(screen.getByText('准备')).toBeTruthy();
        expect(screen.getByText('10')).toBeTruthy();

        act(() => {
            vi.advanceTimersByTime(3000);
        });
        expect(screen.getByText('7')).toBeTruthy();

        click('跳过');
        act(() => {
            vi.advanceTimersByTime(200);
        });
        expect(document.querySelector('.player.work')).toBeTruthy();

        click('暂停');
        const before = document.querySelector('.countdown')!.textContent;
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        expect(document.querySelector('.countdown')!.textContent).toBe(before);
        click('继续');

        click('结束');
        click('修改条件');
        const saved = screen.getByText('我的模板').closest('section')!;
        expect(within(saved).getByText(/^室内-瑜伽垫\+哑铃-30分钟/)).toBeTruthy();
    });

    it('shows a report after finishing and keeps it in the history', () => {
        render(<App />);
        click('15 分钟');
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(20 * 60 * 1000);
        });

        expect(screen.getByText('🎉 训练完成')).toBeTruthy();
        expect(screen.getByText('100%')).toBeTruthy();
        expect(screen.getByText('最近 7 天')).toBeTruthy();
        fireEvent.change(screen.getByLabelText('体重（公斤）'), { target: { value: '60' } });
        click('计算');
        expect(screen.getByText(/^约 \d+ 千卡$/)).toBeTruthy();

        click('返回');
        const history = screen.getByText('训练记录').closest('section')!;
        expect(within(history).getByText(/^室内-瑜伽垫-15分钟/)).toBeTruthy();
        expect(JSON.parse(window.localStorage.getItem('gym.history.v1')!)).toHaveLength(1);
    });

    it('shows a report when ending early after the first minute', () => {
        render(<App />);
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(90 * 1000);
        });
        click('结束');
        expect(screen.getByText(/^训练报告 · 完成 \d+%$/)).toBeTruthy();
        expect(JSON.parse(window.localStorage.getItem('gym.history.v1')!)).toHaveLength(1);
    });

    it('does not record a workout ended within the first minute', () => {
        render(<App />);
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        click('结束');
        expect(screen.getByText('修改条件')).toBeTruthy();
        expect(screen.getByText(/不足 60 秒，没有记录/)).toBeTruthy();
        expect(window.localStorage.getItem('gym.history.v1')).toBeNull();
    });

    it('applies the chosen rest level to the plan', () => {
        render(<App />);
        click('力量');
        click('长一些');
        click('生成训练');
        expect(screen.getByText('组间休息 90 秒')).toBeTruthy();
    });

    it('remembers the coach and voice choice', () => {
        const first = render(<App />);
        click('男教练');
        click('男声');
        first.unmount();

        render(<App />);
        expect(screen.getByRole('button', { name: '男教练' }).getAttribute('aria-pressed')).toBe('true');
        expect(screen.getByRole('button', { name: '男声' }).getAttribute('aria-pressed')).toBe('true');
    });

    it('restores the last conditions on reload', () => {
        const first = render(<App />);
        click('杠铃');
        click('生成训练');
        first.unmount();

        render(<App />);
        expect(screen.getByRole('button', { name: '杠铃' }).getAttribute('aria-pressed')).toBe('true');
    });
});
